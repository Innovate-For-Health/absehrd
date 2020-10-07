WITH DG1 (hadm_id, hf1, long_title)
AS (
	SELECT hadm_id, MAX(CASE WHEN M.icd9_code ILIKE '428%' THEN 1 ELSE 0 END) AS hf1
		, D.long_title
	FROM mimiciii.diagnoses_icd AS M
		JOIN mimiciii.d_icd_diagnoses AS D
			ON M.icd9_code = D.icd9_code 
	WHERE seq_num = 1
	GROUP BY M.hadm_id, D.long_title
)

/* determine if admission had an 'HF, NOS' secondary diagnosis (1) or not (0) */
, DG2 (hadm_id, hf2)
AS (
	SELECT hadm_id, MAX(CASE WHEN icd9_code ILIKE '428%' THEN 1 ELSE 0 END) AS hf2
	FROM mimiciii.diagnoses_icd 
	WHERE seq_num != 1
	GROUP BY hadm_id
)

/* determine if admission had an 'HTN' diagnosis (1) or not (0) */
, DGHTN (hadm_id, htn)
AS (
	SELECT hadm_id,	MAX(CASE WHEN icd9_code ILIKE '401%' THEN 1 ELSE 0 END) AS htn
	FROM mimiciii.diagnoses_icd 
	GROUP BY hadm_id	
)

/* determine if admission had an 'DM2' diagnosis (1) or not (0) */
, DGDM2 (hadm_id, dm2)
AS (
	SELECT hadm_id, MAX(CASE WHEN icd9_code ILIKE '250%' THEN 1 ELSE 0 END) AS dm2
	FROM mimiciii.diagnoses_icd 
	GROUP BY hadm_id
)

/* determine if admission had an 'CKD' diagnosis (1) or not (0) */
, DGCKD (hadm_id, ckd)
AS (
	SELECT hadm_id, MAX(CASE WHEN icd9_code ILIKE '585%' THEN 1 ELSE 0 END) AS ckd
	FROM mimiciii.diagnoses_icd 
	GROUP BY hadm_id
)

/* return total ICU LOS and max SAPSII over all ICU stays for a single admission */
, IESP (hadm_id, total_icu_los, max_sapsii)
AS (
	SELECT IE.hadm_id
		, SUM(IE.los) AS total_icu_los
		, MAX(SP.sapsii) AS max_sapsii
	FROM mimiciii.icustays AS IE
		JOIN mimiciii.sapsii AS SP
			ON IE.icustay_id = SP.icustay_id
	GROUP BY IE.hadm_id
)

/* get average glucose and average sodium lab values for each admission */
, LE (hadm_id, avg_glucose, avg_sodium)
AS (
	SELECT Z.hadm_id, AVG(Z.glucose) AS avg_glucose, AVG(Z.sodium) AS avg_sodium
	FROM (
		SELECT ADM.hadm_id
			, CASE WHEN LE.label = 'Glucose' THEN LE.valuenum ELSE NULL END AS glucose
			, CASE WHEN LE.label = 'Sodium' THEN LE.valuenum ELSE NULL END AS sodium
		FROM mimiciii.admissions AS ADM
			JOIN ( 
					SELECT LE.hadm_id, LD.label, LE.valuenum
					FROM mimiciii.labevents AS LE
						JOIN mimiciii.d_labitems AS LD
							ON LE.itemid = LD.itemid
					WHERE LD.label IN ('Glucose', 'Sodium') AND LD.fluid = 'Blood'
				) AS LE
				ON LE.hadm_id = ADM.hadm_id 
	) AS Z
	GROUP BY Z.hadm_id
)

/* get list of admissions that have at least one ICU stay */
, ICU (hadm_id, n_icustay)
AS (
	SELECT ADM.hadm_id, COUNT(DISTINCT icustay_id) AS n_icustay
	FROM mimiciii.admissions AS ADM
		JOIN mimiciii.icustays AS ICU
			ON ADM.hadm_id = ICU.hadm_id
	GROUP BY ADM.hadm_id
)

, constants (sval, ival)
AS (SELECT '-999999',-999999)

SELECT CASE WHEN SC.ethnicity IS NULL THEN (SELECT sval from constants) ELSE SC.ethnicity END AS ethnicity
	, CASE WHEN SC.gender IS NULL THEN (SELECT sval from constants) ELSE SC.gender END AS gender
	, CASE WHEN SC.age_at_admit IS NULL THEN (SELECT ival from constants) ELSE SC.age_at_admit END AS age_at_admit
	, CASE WHEN SC.n_icustay IS NULL THEN (SELECT ival from constants) ELSE SC.n_icustay END AS n_icustay
	, CASE WHEN SC.max_sapsii IS NULL THEN (SELECT ival from constants) ELSE SC.max_sapsii END AS max_sapsii
	, CASE WHEN SC.total_icu_los IS NULL THEN (SELECT ival from constants) ELSE SC.total_icu_los END AS total_icu_los
	, CASE WHEN SC.avg_glucose IS NULL THEN (SELECT ival from constants) ELSE SC.avg_glucose END AS avg_glucose
	, CASE WHEN SC.avg_sodium IS NULL THEN (SELECT ival from constants) ELSE SC.avg_sodium END AS avg_sodium
	, CASE WHEN SC.hf2 IS NULL THEN (SELECT ival from constants) ELSE SC.hf2 END AS hf2
	, CASE WHEN SC.htn IS NULL THEN (SELECT ival from constants) ELSE SC.htn END AS htn
	, CASE WHEN SC.dm2 IS NULL THEN (SELECT ival from constants) ELSE SC.dm2 END AS dm2
	, CASE WHEN SC.ckd IS NULL THEN (SELECT ival from constants) ELSE SC.ckd END AS ckd
	/*, CASE WHEN SC.died_90d IS NULL THEN (SELECT ival from constants) ELSE SC.died_90d END AS died_90d*/
	, CASE WHEN SC.died_365d IS NULL THEN (SELECT ival from constants) ELSE SC.died_365d END AS died_365d
FROM (
SELECT ADM.ethnicity
		, PAT.gender
		, DATE_PART('year', ADM.admittime) - DATE_PART('year', PAT.dob) AS age_at_admit
		, ICU.n_icustay
		, IESP.max_sapsii
		, IESP.total_icu_los
		, LE.avg_glucose
		, LE.avg_sodium
		, CASE WHEN DG1.hf1 IS NULL THEN 0 ELSE DG1.hf1 END AS hf1
		, CASE WHEN DG2.hf2 IS NULL THEN 0 ELSE DG2.hf2 END AS hf2
		, CASE WHEN DGHTN.htn IS NULL THEN 0 ELSE DGHTN.htn END AS htn
		, CASE WHEN DGDM2.dm2 IS NULL THEN 0 ELSE DGDM2.dm2 END AS dm2
		, CASE WHEN DGCKD.ckd IS NULL THEN 0 ELSE DGCKD.ckd END AS ckd
		/*, CASE WHEN DATE_PART('day', pat.dod - ADM.dischtime) <= 90 THEN 1
			ELSE 0 END AS died_90d*/
		, CASE WHEN DATE_PART('day', pat.dod - ADM.dischtime) <= 365 THEN 1
			ELSE 0 END AS died_365d
		, ADM.hospital_expire_flag
		, ROW_NUMBER() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime ASC) AS adm_num
FROM mimiciii.admissions AS ADM
	LEFT JOIN mimiciii.patients AS PAT ON PAT.subject_id = ADM.subject_id
	LEFT JOIN DG1 ON DG1.hadm_id = ADM.hadm_id
	LEFT JOIN DG2 ON DG2.hadm_id = ADM.hadm_id
	LEFT JOIN DGHTN ON DGHTN.hadm_id = ADM.hadm_id
	LEFT JOIN DGDM2 ON DGDM2.hadm_id = ADM.hadm_id
	LEFT JOIN DGCKD ON DGCKD.hadm_id = ADM.hadm_id
	LEFT JOIN IESP ON IESP.hadm_id = ADM.hadm_id
	LEFT JOIN LE ON LE.hadm_id = ADM.hadm_id
	JOIN ICU ON ICU.hadm_id = ADM.hadm_id
) AS SC
WHERE SC.adm_num = 1	
	AND SC.age_at_admit >= 18
	AND SC.hospital_expire_flag = 0
	AND (SC.hf1 = 1 OR SC.hf2 = 1)

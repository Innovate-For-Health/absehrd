SET search_path TO mimiciii;

/*
-- percentage of patients with valid date of death
-- ~ 33% of patients have a date of death
SELECT COUNT(*)
	, SUM(CASE WHEN dod IS NULL THEN 0 ELSE 1 END) AS N_DEAD
	, SUM(CASE WHEN dod IS NULL THEN 0 ELSE 1 END) * 100.0 / COUNT(*) AS PERC_DEAD
FROM mimiciii.patients
LIMIT 10 
*/


/*
-- do all patients in the cohort have ICU admissions?
-- 46476 unique subject_ids for patients table
-- 46520 unique subject_ids for icustays table
SELECT 'patients', COUNT(DISTINCT subject_id) AS N_PT
FROM mimiciii.patients
UNION 
SELECT 'icustays', COUNT(DISTINCT subject_id) AS N_PT
FROM mimiciii.icustays
--*/


/*
-- do all recorded admissions have associated ICU stays?
-- vast majority of admissions have only 1 icustay associated (54526)
-- only 1190 of admissions have 0 icu stays
SELECT Z.N_ICUSTAY
	, COUNT(Z.N_ICUSTAY) AS N_ADM
FROM (
	SELECT A.hadm_id , COUNT(DISTINCT I.icustay_id) AS N_ICUSTAY
	FROM mimiciii.admissions AS A
		LEFT JOIN mimiciii.icustays AS I
			ON A.hadm_id = I.hadm_id
	GROUP BY A.hadm_id) AS Z
GROUP BY Z.N_ICUSTAY
ORDER BY Z.N_ICUSTAY;
--*/

/*
-- confirm i can calculate who died within a certain time period
SELECT CASE WHEN A.deathtime IS NOT NULL THEN 1 ELSE 0 END AS DIED_ADM
	, CASE WHEN P.dod BETWEEN A.admittime AND (A.dischtime + INTERVAL '30 day') THEN 1 ELSE 0 END AS DIED_MONTH
	, CASE WHEN P.dod BETWEEN A.admittime AND (A.dischtime + INTERVAL '365 day') THEN 1 ELSE 0 END AS DIED_YEAR
	, A.dischtime
	, P.dod
	, DATE_PART('day', P.dod - A.dischtime)
FROM mimiciii.admissions AS A
	JOIN mimiciii.patients AS P
		ON A.subject_id = P.subject_id
LIMIT 100
--*/

/*
-- fraction of admissions resulting in death after a certain time period
-- (admission | month | year) = (9% | 13% | 23%)
-- let's use death during admission as the outcome variable...
SELECT SUM(CASE WHEN A.deathtime IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS PERC_DIED_ADM
	, SUM(CASE WHEN P.dod BETWEEN A.admittime AND (A.dischtime + INTERVAL '30 day') THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS PERC_DIED_MONTH
	, SUM(CASE WHEN P.dod BETWEEN A.admittime AND (A.dischtime + INTERVAL '365 day') THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS PERC_DIED_YEAR
FROM mimiciii.admissions AS A
	JOIN mimiciii.patients AS P
		ON A.subject_id = P.subject_id
--*/

/*
-- is discharge date the same as deathdate if people die during their admission?
-- 99% match... should i filter out those that don't? yes... because it results in LOS
SELECT COUNT(*)
	, SUM(CASE WHEN A.dischtime = A.deathtime THEN 1 ELSE 0 END) AS N_DEATH_EQUALS_DISCH
	, 100.0 * SUM(CASE WHEN A.dischtime = A.deathtime THEN 1 ELSE 0 END) / COUNT(*) AS P_DEATH_EQUALS_DISCH
FROM mimiciii.admissions AS A
WHERE A.deathtime IS NOT NULL
--*/

/*
-- what is the LOS for patients who die during their admissions
SELECT DATE_PART('day', A.dischtime - A.admittime) * 24 + DATE_PART('hour', A.dischtime - A.admittime) AS LOS_HOUR
	, DATE_PART('day', A.dischtime - A.admittime) + DATE_PART('hour', A.dischtime - A.admittime) / 24.0 AS LOS_DAY
FROM mimiciii.admissions AS A
WHERE A.deathtime IS NOT NULL
	AND A.dischtime >= A.admittime
--*/

/*
CREATE OR REPLACE FUNCTION _final_median(NUMERIC[])
   RETURNS NUMERIC AS
$$
   SELECT AVG(val)
   FROM (
     SELECT val
     FROM unnest($1) val
     ORDER BY 1
     LIMIT  2 - MOD(array_upper($1, 1), 2)
     OFFSET CEIL(array_upper($1, 1) / 2.0) - 1
   ) sub;
$$
LANGUAGE 'sql' IMMUTABLE;
 
CREATE AGGREGATE median(NUMERIC) (
  SFUNC=array_append,
  STYPE=NUMERIC[],
  FINALFUNC=_final_median,
  INITCOND='{}'
);
--*/
/*
-- what is average LOS for patients who die during their admissions
SELECT round(MIN(Z.LOS_DAY),2) AS MIN_LOS_DAY
	, round(median(Z.LOS_DAY),2) AS MEDIAN_LOS_DAY
	, round(AVG(Z.LOS_DAY),2) AS AVG_LOS_DAY
	, round(MAX(Z.LOS_DAY),2) AS MAX_LOS_DAY
FROM (
SELECT CAST(DATE_PART('day', A.dischtime - A.admittime) + DATE_PART('hour', A.dischtime - A.admittime) / 24.0 AS NUMERIC) AS LOS_DAY
FROM mimiciii.admissions AS A
WHERE A.deathtime IS NOT NULL
	AND A.dischtime >= A.admittime) AS Z;
--*/

/* do all admissions originate from the ED? no */
/*
SELECT *
FROM admissions
LIMIT 10;
--*/

/* get the primary diagnosis code for the most recent previous discharge within a year of the index admission */
/*
SELECT Z.*
FROM (
SELECT A1.hadm_id, A2.hadm_id, DD.icd9_code, DD.short_title, DD.long_title
	, ROW_NUMBER() OVER (PARTITION BY A1.hadm_id ORDER BY A2.dischtime DESC) AS RN
FROM admissions AS A1
	JOIN admissions AS A2
		ON A1.subject_id = A2.subject_id AND A1.hadm_id != A2.hadm_id 
			AND A2.admittime BETWEEN A1.dischtime - INTERVAL '365day' AND A1.admittime
	JOIN diagnoses_icd AS D2
		ON A2.hadm_id = D2.hadm_id
	JOIN d_icd_diagnoses AS DD
		ON D2.icd9_code = DD.icd9_code
WHERE D2.seq_num = 1) AS Z
WHERE Z.RN = 1
LIMIT 10
--*/

/* data mart for predicting mortality during admission */
-- what about diagnosis codes from last admission? if there is a last admission?  m
--/*
SELECT A.hadm_id
	, CASE WHEN A.deathtime IS NOT NULL THEN 1 ELSE 0 END AS died_adm
	, P.gender
	, DATE_PART('year', A.admittime) - DATE_PART('year', P.dob) AS age
	, A.marital_status
	, A.insurance
	, A.ethnicity
	, A.language
	, A.admission_type
	, A.admission_location
	, CASE WHEN I30.N_ICU_30_DAY IS NULL THEN 0 ELSE I30.N_ICU_30_DAY END AS N_ICU_30_DAY
	, CASE WHEN I365.N_ICU_365_DAY IS NULL THEN 0 ELSE I365.N_ICU_365_DAY END AS N_ICU_365_DAY
	, CASE WHEN A30.N_ADM_30_DAY IS NULL THEN 0 ELSE A30.N_ADM_30_DAY END AS N_ADM_30_DAY
	, CASE WHEN A365.N_ADM_365_DAY IS NULL THEN 0 ELSE A365.N_ADM_365_DAY END AS N_ADM_365_DAY
	, D.icd9_code AS PREV_ADM_DIAG_CODE
	, D.short_title AS PREV_ADM_DIAG_TITLE
FROM admissions AS A
	JOIN patients AS P
		ON A.subject_id = P.subject_id
	LEFT JOIN 
		(SELECT A.hadm_id, COUNT(DISTINCT I.icustay_id) AS N_ICU_30_DAY
		 FROM admissions AS A 
		 	JOIN icustays AS I
				ON A.subject_id = I.subject_id AND I.outtime BETWEEN A.admittime - INTERVAL '30day' AND A.admittime
		 GROUP BY A.hadm_id
		 ) AS I30
		 	ON I30.hadm_id = A.hadm_id
	LEFT JOIN 
		(SELECT A.hadm_id, COUNT(DISTINCT I.icustay_id) AS N_ICU_365_DAY
		 FROM admissions AS A 
		 	JOIN icustays AS I
				ON A.subject_id = I.subject_id AND I.outtime BETWEEN A.admittime - INTERVAL '365day' AND A.admittime
		 GROUP BY A.hadm_id
		 ) AS I365
		 	ON I365.hadm_id = A.hadm_id
	LEFT JOIN 
		(SELECT A1.hadm_id, COUNT(DISTINCT A2.hadm_id) AS N_ADM_30_DAY
		 FROM admissions AS A1
		 	JOIN admissions AS A2
				ON A1.subject_id = A2.subject_id AND A1.hadm_id != A2.hadm_id 
		 			AND A2.admittime BETWEEN A1.dischtime - INTERVAL '30day' AND A1.admittime
		 GROUP BY A1.hadm_id
		 ) AS A30
		 	ON A30.hadm_id = A.hadm_id
	LEFT JOIN 
		(SELECT A1.hadm_id, COUNT(DISTINCT A2.hadm_id) AS N_ADM_365_DAY
		 FROM admissions AS A1
		 	JOIN admissions AS A2
				ON A1.subject_id = A2.subject_id AND A1.hadm_id != A2.hadm_id 
		 			AND A2.admittime BETWEEN A1.dischtime - INTERVAL '365day' AND A1.admittime
		 GROUP BY A1.hadm_id
		 ) AS A365
		 	ON A365.hadm_id = A.hadm_id
	LEFT JOIN 
		(SELECT Z.*
			FROM (
			SELECT A1.hadm_id, DD.icd9_code, DD.short_title, DD.long_title
				, ROW_NUMBER() OVER (PARTITION BY A1.hadm_id ORDER BY A2.dischtime DESC) AS RN
			FROM admissions AS A1
				JOIN admissions AS A2
					ON A1.subject_id = A2.subject_id AND A1.hadm_id != A2.hadm_id 
						AND A2.admittime BETWEEN A1.dischtime - INTERVAL '365day' AND A1.admittime
				JOIN diagnoses_icd AS D2
					ON A2.hadm_id = D2.hadm_id
				JOIN d_icd_diagnoses AS DD
					ON D2.icd9_code = DD.icd9_code
			WHERE D2.seq_num = 1) AS Z
			WHERE Z.RN = 1
			) AS D
			ON D.hadm_id = A.hadm_id
LIMIT 100
--*/


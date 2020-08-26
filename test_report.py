import numpy as np
from report import report

class tester_rep(object):
    
    def test_prediction_report():
        
        file_pred = "../plots/test1_prediction_report.pdf"
        n = 1000
        m = 7
        col_names=np.array(['A','B','C','D','E','F','G'])
        outcome ='F'
        r = np.random.randint(low=0, high=2, size=(n, m))
        s = np.random.randint(low=0, high=2, size=(n, m))
        
        res1 = report.prediction_report(report, 
                                       r=r, 
                                       s=s, 
                                       col_names=col_names, 
                                       outcome=outcome, 
                                       file_pdf=file_pred)
        
        file_pred = "../plots/test2_prediction_report.pdf"
        n = 1000
        m = 7
        col_names=np.array(['A','B','C','D','E','F','G'])
        outcome ='F'
        r = np.column_stack((np.random.uniform(low=0, high=1, size=(n, m-1)), 
                         np.random.randint(low=0, high=2, size=n)))
        s = np.column_stack((np.random.uniform(low=0, high=1, size=(n, m-1)), 
                         np.random.randint(low=0, high=2, size=n)))
        
        res2 = report.prediction_report(report, 
                                       r=r, 
                                       s=s, 
                                       col_names=col_names, 
                                       outcome=outcome, 
                                       file_pdf=file_pred)
        
        return res1 and res2
    
    def test_description_report():
        
        file_desc = "../plots/test_description_report.pdf"
        n = 1000
        m = 7
        col_names=np.array(['A','B','C','D','E','F','G'])
        outcome ='G'
        r = np.random.randint(low=0, high=2, size=(n, m))
        s = np.random.randint(low=0, high=2, size=(n, m))
        
        res_desc = report.description_report(report, 
                                       r=r, 
                                       s=s, 
                                       col_names=col_names, 
                                       outcome=outcome, 
                                       file_pdf=file_desc)
        
        return res_desc
    
def main():
    
    buffer = "\t\t"
    
    print('Testing report.prediction_report', end='...' + buffer)
    print('PASS') if tester_rep.test_prediction_report() else print('FAIL')
    
    print('Testing report.description_report', end='...' + buffer)
    print('PASS') if tester_rep.test_description_report() else print('FAIL')
        
if __name__ == "__main__":
    main()


    
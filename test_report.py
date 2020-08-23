import numpy as np
from report import report

class tester_rep(object):
    
    def test_prediction_report():
        
        file_pred = "../plots/test_prediction_report.pdf"
        n = 1000
        m = 7
        col_names=np.array(['A','B','C','D','E','F','G'])
        outcome ='F'
        r = np.random.randint(low=0, high=2, size=(n, m))
        s = np.random.randint(low=0, high=2, size=(n, m))
        
        res_pred = report.prediction_report(report, 
                                       r=r, 
                                       s=s, 
                                       col_names=col_names, 
                                       outcome=outcome, 
                                       file_pdf=file_pred)
        
        return res_pred
    
    def test_description_report():
        
        file_desc = "../plots/test_description_report.pdf"
        n = 1000
        m = 7
        col_names=np.array(['A','B','C','D','E','F','G'])
        outcome ='F'
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


    
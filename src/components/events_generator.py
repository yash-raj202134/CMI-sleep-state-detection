import pandas as pd

def get_events(preds, probs, test, prediction_window=10+1, score_window=10+1) :
    test.loc[:, 'prediction'] = preds
    test.loc[:, 'prediction'] = test['prediction'].rolling(prediction_window, center=True).median()
    test.loc[:, 'probability'] = probs
    
    # test.loc[test['prediction']==0, 'probability'] = 1-test.loc[test['prediction']==0, 'probability']
    test.loc[:, 'score'] = test['probability'].rolling(score_window, center=True, min_periods=10).mean().bfill().ffill()

    test.loc[:, 'pred_diff'] = test['prediction'].diff()
    
    test.loc[:, 'event'] = test['pred_diff'].replace({1:'wakeup', -1:'onset', 0:np.nan})
    
    test_wakeup = test[test['event']=='wakeup'].groupby(test['timestamp'].dt.date).agg('first')
    test_onset = test[test['event']=='onset'].groupby(test['timestamp'].dt.date).agg('last')
    test = pd.concat([test_wakeup, test_onset], ignore_index=True).sort_values('timestamp')
    
    test['step'] = test['step'].astype('int32')

    return test

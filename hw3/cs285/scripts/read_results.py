import glob
import tensorflow as tf

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'TimeSinceStart':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
    return X, Y

def get_avg_returns(logdirs):
    avg_returns = []
    for logdir in logdirs:
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results(eventfile)
        avg_returns.append(Y)
        
    return avg_returns    

if __name__ == '__main__':
    import glob

    double_dqn_logdirs = []
    dqn_logdirs = []
    for i in [1, 2, 3]:
        double_dqn = f'data/q2_doubledqn_{i}_LunarLander-v3_17-03-2023_23-23-21/events*'
        dqn = f'data/q2_dqn_{i}_LunarLander-v3_17-03-2023_23-23-21/events*'
        double_dqn_logdirs.append(double_dqn)
        dqn_logdirs.append(dqn)
    
    double_dqn_returns = get_avg_returns(double_dqn_logdirs)
    dqn_returns = get_avg_returns(dqn_logdirs)        


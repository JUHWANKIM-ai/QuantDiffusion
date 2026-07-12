from src.evaluation.eval_portfolio.loss import *
from src.evaluation.eval_portfolio.strategies import *
from src.utils import *

STRATEGIES ={
             'equal_weight': EqualWeightPortfolioStrategy,             
             'mean_reversion': MeanReversionStrategy,
             'trend_following': TrendFollowingStrategy,
             'vol_trading': VolatilityTradingStrategy
             }

def full_evaluation(fake_dataset, real_dataset, config, **kwargs):
    ec = EvaluationComponent(config, fake_dataset, real_dataset, **kwargs)
    summary_dict = ec.eval_summary()    
    return summary_dict, ec.pnl_real_npy, ec.pnl_fake_npy

class EvaluationComponent(object):
    '''
    Evaluation component for evaluation metrics according to config
    '''
    def __init__(self, config, fake_dataset, real_data, **kwargs):
        self.config = config
        self.real_data = real_data
        self.fake_data = fake_dataset
        self.kwargs = kwargs
        self.n_eval = self.config.Evaluation.n_eval
        self.sample_size = min(self.real_data.shape[0], self.config.Evaluation.batch_size)
        
        self.pnl_real_npy = None
        self.pnl_fake_npy = None

        self.seed = kwargs.get('seed', getattr(config, 'seed', 52))
        set_seed(self.seed)
        
        self.data_set = self.get_data(n=self.n_eval)

        strat_name = kwargs.get('strat_name', 'equal_weight')
        self.strat = STRATEGIES[strat_name]()
        
        self.metrics_group = {            
            'tail_scores': ['var', 'es'],
            'trading_strat_scores': ['max_drawback', 'cumulative_pnl']
        }

    def get_data(self, n=1):
        batch_size = int(self.sample_size)
        idx_all = torch.randperm(self.real_data.shape[0])[:batch_size * n]        
        data = {}
        for i in range(n):
            idx = idx_all[i * batch_size:(i + 1) * batch_size]            
            data[i] = {
                'real': self.real_data[idx],
                'fake': self.fake_data[idx]
            }
        return data

    def eval_summary(self):
        assert self.n_eval == 1, "This version only supports n_eval == 1"

        metrics = self.config.Evaluation.metrics_enabled        
        summary = {}

        # 1. 데이터
        real = self.data_set[0]['real']
        fake = self.data_set[0]['fake']
        
        # 2. 전략별 PnL 계산               
        pnl_real = self.strat.get_pnl_trajectory(real)
        pnl_fake = self.strat.get_pnl_trajectory(fake)
        
        self.pnl_real_npy = to_numpy(pnl_real.clone())
        self.pnl_fake_npy = to_numpy(pnl_fake.clone())
        
        # 3. metric 계산
        for grp in self.metrics_group:
            for metric in [m for m in metrics if m in self.metrics_group[grp]]:
                eval_func = getattr(self, metric)
                abs_loss_tensor, rel_loss_tensor = eval_func(pnl_real, pnl_fake)
                abs_loss = to_numpy(abs_loss_tensor).mean()
                rel_loss = to_numpy(rel_loss_tensor).mean()
                summary[f'{metric}_abs_mean'] = abs_loss
                summary[f'{metric}_rel_mean'] = rel_loss

        return summary
    
    def var(self, real, fake):
        ecfg = self.config.Evaluation.TestMetrics.var    
        losses = VARLoss(real.unsqueeze(2), name='var_loss', alpha=ecfg.alpha)(fake.unsqueeze(2))        
        abs_loss = losses['abs_loss']
        rel_loss = losses['rel_loss']  
        return (abs_loss, rel_loss)
    
    def es(self, real, fake):
        ecfg = self.config.Evaluation.TestMetrics.es   
        losses = ESLoss(real.unsqueeze(2), name='es_loss', alpha=ecfg.alpha)(fake.unsqueeze(2))        
        abs_loss = losses['abs_loss']
        rel_loss = losses['rel_loss']  
        return (abs_loss, rel_loss)

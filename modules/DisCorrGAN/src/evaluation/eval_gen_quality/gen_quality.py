import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import acf, ccf
from scipy.stats import wasserstein_distance, ks_2samp, skew, kurtosis
from src.utils import rolling_window


def calculate_distribution_scores(real, fake, n_vars, windows):
    windows = pd.Series(windows, name='window size')
    scores = {
        'EMD': np.zeros((n_vars, len(windows))),
        'KS': np.zeros((n_vars, len(windows))),
    }

    for i in range(n_vars):
        for j in range(len(windows)):
            real_dist = rolling_window(real[i].T, windows[j]).sum(axis=1).ravel()
            fake_dist = rolling_window(fake[i].T, windows[j]).sum(axis=1).ravel()
                        
            scores['EMD'][i, j] = wasserstein_distance(real_dist, fake_dist)                    
            scores['KS'][i, j], _ = ks_2samp(real_dist, fake_dist)                                    
                
    df_scores = {}
    for metric, data in scores.items():
        data = np.round(data, decimals=4)
        df_scores[metric] = pd.DataFrame(data.T, index=windows, columns=[f'{metric} {i}' for i in range(n_vars)])
    
    emd_avg = np.mean(scores['EMD'], axis=0)
    ks_avg = np.mean(scores['KS'], axis=0)
    df_scores['EMD']['EMD_avg'] = np.round(emd_avg, decimals=4)    
    df_scores['KS']['KS_avg'] = np.round(ks_avg, decimals=4)

    return df_scores

def plot_distribution_comparison(real_list, fake_list, n_vars, windows):
    for j in range(n_vars):
        fig, axs = plt.subplots(nrows=1, ncols=len(windows), figsize=(5.5 * len(windows), 4))  

        for i, window in enumerate(windows):
            real_dist = rolling_window(real_list[j].T, window).sum(axis=1).ravel()
            fake_dist = rolling_window(fake_list[j].T, window).sum(axis=1).ravel()

            min_val = real_dist.min()
            max_val = real_dist.max()
            bins = np.linspace(min_val, max_val, 81)

            sns.histplot(real_dist, bins=bins, kde=False, ax=axs[i], color='tab:blue', linewidth=0.1, alpha=0.5, stat='density')
            sns.histplot(fake_dist, bins=bins, kde=False, ax=axs[i], color='tab:orange', linewidth=0.1, alpha=0.5, stat='density')

            axs[i].set_xlim(*np.quantile(real_dist, [0.001, 0.999]))
            axs[i].yaxis.grid(True, alpha=0.5)
            axs[i].set_xlabel('Cumulative log return', fontsize=12)
            axs[i].set_ylabel('Density', fontsize=12)
            #axs[i].set_title(f'{window}-day return', fontsize=14)

        axs[0].legend(['Real returns', 'Synthetic returns'])
        plt.savefig(f"./outputs/figure_dist{j}.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

def plot_distribution_comparison_vertical(real_list, fake_list, n_vars, windows):
    for j in range(n_vars):
        fig, axs = plt.subplots(nrows=len(windows), ncols=1, figsize=(6, 5 * len(windows)))  

        for i in range(len(windows)):
            window = windows[i]

            real_dist = rolling_window(real_list[j].T, window).sum(axis=1).ravel()
            fake_dist = rolling_window(fake_list[j].T, window).sum(axis=1).ravel()

            min_val = real_dist.min()
            max_val = real_dist.max()
            bins = np.linspace(min_val, max_val, 81)

            sns.histplot(real_dist, bins=bins, kde=False, ax=axs[i], color='tab:blue', linewidth=0.1, alpha=0.5, stat='density')
            sns.histplot(fake_dist, bins=bins, kde=False, ax=axs[i], color='tab:orange', linewidth=0.1, alpha=0.5, stat='density')

            axs[i].set_xlim(*np.quantile(real_dist, [0.001, 0.999]))
            axs[i].yaxis.grid(True, alpha=0.5)
            axs[i].set_xlabel('Cumulative log return', fontsize=12)
            axs[i].set_ylabel('Frequency', fontsize=12)
            axs[i].set_title(f'{window}-day return', fontsize=14)

        axs[0].legend(['Real returns', 'Synthetic returns'])
        plt.savefig(f"./outputs/figure_dist_vertical{j}.png", dpi=300, bbox_inches='tight')
        plt.close()

def calculate_skew_kurtosis(real_list, fake_list, windows):
    num_assets = len(real_list)
    asset_cols = [f"Asset{i+1}" for i in range(num_assets)]

    df_skew = pd.DataFrame(index=windows, columns=asset_cols, dtype=float)
    df_kurt = pd.DataFrame(index=windows, columns=asset_cols, dtype=float)

    for w in windows:
        for i in range(num_assets):
            real_arr = real_list[i] 
            fake_arr = fake_list[i]
            
            real_skews, real_kurts = [], []
            fake_skews, fake_kurts = [], []

            num = min(real_arr.shape[0], fake_arr.shape[0])
            for k in range(num): 
                real_win = rolling_window(real_arr[k:k+1].T, w).sum(axis=1).ravel()
                fake_win = rolling_window(fake_arr[k:k+1].T, w).sum(axis=1).ravel()
                
                real_skews.append( skew(real_win) )
                real_kurts.append( kurtosis(real_win, fisher=False) )
                fake_skews.append( skew(fake_win) )
                fake_kurts.append( kurtosis(fake_win, fisher=False) )

            avg_real_sk = np.mean(real_skews)
            avg_fake_sk = np.mean(fake_skews)
            avg_real_ku = np.mean(real_kurts)
            avg_fake_ku = np.mean(fake_kurts)

            df_skew.at[w,  asset_cols[i]] = abs(avg_real_sk - avg_fake_sk)
            df_kurt.at[w, asset_cols[i]] = abs(avg_real_ku - avg_fake_ku)

        df_skew.at[w,  "Avg"] = df_skew.loc[w,  asset_cols].mean()
        df_kurt.at[w, "Avg"] = df_kurt.loc[w, asset_cols].mean()

    return df_skew.round(4), df_kurt.round(4)



def plot_acf_comparison(real_list, fake_list, n_vars, lags):
    data_transforms = [
        (lambda x: x,       'Identity', 'Identity log returns'),
        (np.abs,            'Absolute', 'Absolute log returns'),
        (np.square,         'Squared',  'Squared log returns')
    ]
    
    for i in range(n_vars):
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
        
        for ax, (transform_func, data_type, title) in zip(axs, data_transforms):
            transformed_real = transform_func(real_list[i])
            transformed_fake = transform_func(fake_list[i])
            
            acf_real = np.array([acf(ts, nlags=lags) for ts in transformed_real])
            acf_fake = np.array([acf(ts, nlags=lags) for ts in transformed_fake])
            
            mean_real = acf_real.mean(axis=0)
            std_real  = acf_real.std(axis=0)
            mean_fake = acf_fake.mean(axis=0)
            std_fake  = acf_fake.std(axis=0)
            
            ax.plot(mean_real, label=f'Real (mean)', color='tab:blue')
            ax.fill_between(range(lags+1),
                            mean_real - 0.5*std_real,
                            mean_real + 0.5*std_real,
                            color='tab:blue', alpha=0.2,
                            label='Real ± 1/2 std')
            
            ax.plot(mean_fake, label=f'Synthetic (mean)', color='tab:orange')
            ax.fill_between(range(lags+1),
                            mean_fake - 0.5*std_fake,
                            mean_fake + 0.5*std_fake,
                            color='tab:orange', alpha=0.2,
                            label='Synthetic ± 1/2 std')
            
            ax.set_ylim(-0.15, 0.3)
            #ax.set_title(title, fontsize=13)
            ax.grid(True)
            ax.axhline(y=0, color='k', linewidth=0.8)
            ax.axvline(x=0, color='k', linewidth=0.8)
            ax.set_xlabel('Lag', fontsize=12)
        
        axs[0].legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"./outputs/figure_acf{i}.png", dpi=300, bbox_inches='tight')
        plt.show()

def plot_acf_comparison_vertical(real_list, fake_list, n_vars, lags=50):
    data_transforms = [
        (lambda x: x,       'Identity', 'Identity log returns'),
        (np.abs,            'Absolute', 'Absolute log returns'),
        (np.square,         'Squared',  'Squared log returns')
    ]
    
    for i in range(n_vars):
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 12))
        
        for ax, (transform_func, data_type, title) in zip(axs, data_transforms):
            transformed_real = transform_func(real_list[i])
            transformed_fake = transform_func(fake_list[i])
            
            acf_real = np.array([acf(ts, nlags=lags) for ts in transformed_real])
            acf_fake = np.array([acf(ts, nlags=lags) for ts in transformed_fake])
            
            mean_real = acf_real.mean(axis=0)
            std_real  = acf_real.std(axis=0)
            mean_fake = acf_fake.mean(axis=0)
            std_fake  = acf_fake.std(axis=0)
            
            ax.plot(mean_real, label=f'Real (mean)', color='tab:blue')
            ax.fill_between(range(lags+1),
                            mean_real - 0.5*std_real,
                            mean_real + 0.5*std_real,
                            color='tab:blue', alpha=0.2,
                            label='Real ± 1/2 std')
            
            ax.plot(mean_fake, label=f'Synthetic (mean)', color='tab:orange')
            ax.fill_between(range(lags+1),
                            mean_fake - 0.5*std_fake,
                            mean_fake + 0.5*std_fake,
                            color='tab:orange', alpha=0.2,
                            label='Synthetic ± 1/2 std')
            
            ax.set_ylim(-0.15, 0.3)
            ax.set_title(title, fontsize=13)
            ax.grid(True)
            ax.axhline(y=0, color='k', linewidth=0.8)
            ax.axvline(x=0, color='k', linewidth=0.8)
            ax.set_xlabel('Lag', fontsize=12)
            ax.set_ylabel('ACF', fontsize=12)
        
        axs[0].legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"./outputs/figure_acf_vertical{i}.png", dpi=300, bbox_inches='tight')
        #plt.show()
        plt.close()
        
        
def calculate_acf_score(real_list, fake_list, lags, loss_type):

    data_transforms = [lambda x: x, np.abs, np.square]  
    titles = ['Identity log returns', 'Absolute log returns', 'Squared log returns']

    acf_scores = {} 
    avg_scores_per_transform = {title: [] for title in titles}

    n_vars = len(real_list)
    for group_idx in range(n_vars):
        group_scores = {} 

        for transform, title in zip(data_transforms, titles):
            transformed_real = transform(real_list[group_idx])
            transformed_fake = transform(fake_list[group_idx])
            
            acf_real = np.array([acf(ts, nlags=lags) for ts in transformed_real])  
            acf_fake = np.array([acf(ts, nlags=lags) for ts in transformed_fake])  

            mean_acf_real = np.mean(acf_real, axis=0)  
            mean_acf_fake = np.mean(acf_fake, axis=0) 

            diff = mean_acf_fake - mean_acf_real

            if loss_type == 'mse':
                val = np.mean(diff**2)
            elif loss_type == 'mae':
                val = np.mean(np.abs(diff))
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            group_scores[title] = round(val, 4)

            avg_scores_per_transform[title].append(val)

        acf_scores[f"Group {group_idx+1}"] = group_scores
    
    overall_average_scores = {}
    for title in titles:
        mean_of_means = np.mean(avg_scores_per_transform[title])
        overall_average_scores[title] = round(mean_of_means, 4)

    return acf_scores, overall_average_scores


def ccf_mean(data):
    num_samples = data.shape[0]
    corr = []
    for i in range(num_samples):
        sample = data[i]  
        corr_mat = np.corrcoef(sample, rowvar=False)
        corr.append(corr_mat)
    mean_corr_matrix = np.mean(corr, axis=0)  
    return mean_corr_matrix

def ccf_loss(real_mean_corr, fake_mean_corr, loss_type):
    diff = fake_mean_corr - real_mean_corr
    if loss_type == 'mse':
        loss_val = np.mean(diff**2)
    elif loss_type == 'mae':
        loss_val = np.mean(np.abs(diff))
    elif loss_type == 'frobenius':            
        loss_val = np.linalg.norm(diff, ord='fro')
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    return round(loss_val, 4)


def plot_ccf_heatmap(corr_matrix, title, annot=True):
    fig, ax = plt.subplots(figsize=(3.5, 3))

    heatmap_kwargs = dict(
        data=corr_matrix,
        cmap='coolwarm',
        vmin=-0.1,
        vmax=1,
        linewidths=0.01,
        linecolor="white",
        ax=ax,
        annot=annot,
        fmt=".2f" if annot else ""
    )
    
    if annot:
        heatmap_kwargs["annot_kws"] = {"size": 12}
    sns.heatmap(**heatmap_kwargs)

    # ax.set_title(title, fontsize=15)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"./outputs/fig_corr_{title}.png", dpi=300, bbox_inches='tight')

    plt.show()
    
    
def ccf_lag_single_sample(x, lag):
    time_steps, num_features = x.shape
    cc_matrix = np.zeros((num_features, num_features, lag+1), dtype=float)

    for i in range(num_features):
        for j in range(num_features):
            c_array = ccf(x[:, i], x[:, j], adjusted=False)
            c_array = c_array[:(lag+1)]
            if len(c_array) < (lag+1):
                c_array = np.pad(c_array, (0, lag+1 - len(c_array)), 'constant')
            cc_matrix[i, j, :] = c_array

    return cc_matrix

def ccf_lag_mean(data, lag):
    num_samples, time_steps, num_features = data.shape
    all_cc = np.zeros((num_samples, num_features, num_features, lag+1), dtype=float)

    for s_idx in range(num_samples):
        cc_matrix = ccf_lag_single_sample(data[s_idx], lag=lag)
        all_cc[s_idx] = cc_matrix 

    mean_ccf = all_cc.mean(axis=0) 
    return mean_ccf

def ccf_lag_loss(real_mean_ccf, fake_mean_ccf, loss_type='mse'):
    diff = fake_mean_ccf - real_mean_ccf

    if loss_type == 'mse':
        loss_val = np.mean(diff**2)
    elif loss_type == 'mae':
        loss_val = np.mean(np.abs(diff))
    elif loss_type == 'frobenius':
        loss_val = np.linalg.norm(diff)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    return round(loss_val, 4)


def leverage_effect_loss(real_list, fake_list, lags, loss_type):
    def leverage_effect(ts, tau):
        rt = ts[:-tau]            
        rt_squared = ts[tau:]**2 
        return np.corrcoef(rt, rt_squared)[0, 1]

    n_vars = len(real_list)
    leverage_scores = {}
    all_group_losses = []

    for i in range(n_vars):
        real_data = real_list[i] 
        fake_data = fake_list[i] 

        real_leverage_effects = np.array([
            [leverage_effect(ts, tau) for tau in range(1, lags+1)]
            for ts in real_data
        ])
        
        fake_leverage_effects = np.array([
            [leverage_effect(ts, tau) for tau in range(1, lags+1)]
            for ts in fake_data
        ])
        
        mean_real_leverage = np.mean(real_leverage_effects, axis=0)         
        mean_fake_leverage = np.mean(fake_leverage_effects, axis=0) 
        
        diff = mean_fake_leverage - mean_real_leverage
        
        if loss_type == 'mse':
            loss_val = np.mean(diff**2)
        elif loss_type == 'mae':
            loss_val = np.mean(np.abs(diff))
        elif loss_type == 'frobenius':
            loss_val = np.linalg.norm(diff)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        
        leverage_scores[f'Group {i+1}'] = loss_val
        all_group_losses.append(loss_val)

        print(f"[Group {i+1}] Leverage Loss ({loss_type.upper()}) => {loss_val:.4f}")
            
    overall_mean = np.mean(all_group_losses)
    print(f"\n[Overall] Leverage Loss ({loss_type.upper()}) => {overall_mean:.4f}")
    
    for group in leverage_scores:
        leverage_scores[group] = round(leverage_scores[group], 4)
    overall_mean = round(overall_mean, 4)

    return leverage_scores, overall_mean


def plot_best_generated_sample(fake_data, real_mean_corr, asset_labels):
    sns.set_style("darkgrid")
    
    n_assets = fake_data.shape[1]
    custom_palette = sns.color_palette("Dark2", n_assets)
    
    mae_values = []
    for i in range(fake_data.shape[0]):
        fake_sample = fake_data[i] 
        fake_corr = np.corrcoef(fake_sample)
        mae = np.mean(np.abs(fake_corr - real_mean_corr))
        mae_values.append(mae)

    best_sample_idx = np.argmin(mae_values)
    best_sample_mae = mae_values[best_sample_idx]
    best_sample = fake_data[best_sample_idx]
    
    cumulative_returns = np.cumsum(best_sample, axis=1)
    cumulative_returns = np.hstack((np.zeros((n_assets, 1)), cumulative_returns))

    plt.figure(figsize=(12, 6))
    for asset_idx in range(n_assets):
        plt.plot(cumulative_returns[asset_idx], linewidth=1.5, label=asset_labels[asset_idx], color=custom_palette[asset_idx])

    plt.xlabel("T (number of days)", fontsize=14)
    plt.ylabel("Cumulative Log Return", fontsize=14)
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig(f'./outputs/figure_MTS_fake_sample.png', dpi=300, bbox_inches='tight')
    print(f"Best Sample MAE: {best_sample_mae:.4f}")
    plt.show()


def plot_real_cumulative_log_returns(df, asset_labels, filename='cumulative_log_returns_real.png'):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("darkgrid")

    n_assets = len(asset_labels)
    custom_palette = sns.color_palette("Dark2", n_assets)

    log_returns = np.log(df / df.shift(1)).dropna()
    cumulative_log_returns = np.cumsum(log_returns.values, axis=0)
    cumulative_log_returns = np.vstack((np.zeros((1, n_assets)), cumulative_log_returns))

    plt.figure(figsize=(12, 6))
    for asset_idx in range(n_assets):
        plt.plot(
            df.index[:len(cumulative_log_returns)-1],
            cumulative_log_returns[:-1, asset_idx],
            linewidth=1.5,
            label=asset_labels[asset_idx],
            color=custom_palette[asset_idx]
        )

    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Log Return", fontsize=14)
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig(f'./outputs/{filename}', dpi=300, bbox_inches='tight')
    plt.show()

def plot_cumulative_pnl_quantiles(pnl_real, pnl_fake, strat_name):

    sorted_real = np.sort(pnl_real[:, 20:], axis=0)
    sorted_fake = np.sort(pnl_fake[:, 20:], axis=0)

    mean_real_by_rank = sorted_real.mean(axis=1)[:-1]
    mean_fake_by_rank = sorted_fake.mean(axis=1)[:-1]
    
    np.savez(
        f"./outputs/cumulative_pnl_quantiles_{strat_name}.npz",
        mean_real=mean_real_by_rank,
        mean_fake=mean_fake_by_rank
    )

    num_ranks = len(mean_real_by_rank)
    ranks = np.arange(1, num_ranks + 1) / num_ranks

    plt.figure(figsize=(6, 5))
    plt.plot(ranks, mean_real_by_rank, label=f"{strat_name} (real)", linewidth=2)
    plt.plot(ranks, mean_fake_by_rank, linestyle='--', label=f"{strat_name} (fake)", linewidth=2)

    plt.xscale('log')  

    plt.xlabel('α (Normalized Rank, log scale)', fontsize=12)
    plt.ylabel('PnL', fontsize=12)
    #plt.title(f'Cumulative_PnL_Quantile: {strat_name}', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
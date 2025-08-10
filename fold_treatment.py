import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import os
import argparse


def analyze_fold_summary(fold_summary_path, model_type, output_json=True):
    with open(fold_summary_path) as f:
        folds = json.load(f)

    MIN_POS, MAX_POS = 0.10, 0.60   
    MIN_VOL = 1e-4                    
    MIN_FOLDS_PER_TICKER = 2          
    USE_IQR_OUTLIERS = True

    fig_dir = os.path.join('data', 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    ratios, vols, tickers, fold_ids = [], [], [], []
    kept, dropped = [], []

    def read_ratio_from_meta(fold):
        meta_path = os.path.join('data/processed_folds', f'{model_type}_meta',
                                 f'val_meta_fold_{fold["global_fold_id"]}.csv')
        if not os.path.exists(meta_path):
            return None
        dfm = pd.read_csv(meta_path)
        return float(dfm['target'].mean()) if 'target' in dfm.columns else None

    def read_vol_from_meta(fold):
        meta_path = os.path.join('data/processed_folds', f'{model_type}_meta',
                                 f'val_meta_fold_{fold["global_fold_id"]}.csv')
        if not os.path.exists(meta_path):
            return None
        dfm = pd.read_csv(meta_path)
        return float(dfm['Log_Returns'].std()) if 'Log_Returns' in dfm.columns else None

    for fold in folds:
        gid = fold.get('global_fold_id')
        ticker = fold.get('ticker')

        ratio = fold.get('val_pos_ratio') or read_ratio_from_meta(fold)
        if ratio is None:
            dropped.append({**fold, 'drop_reason': 'missing_target_or_meta'})
            continue

        vol = fold.get('val_vol') or read_vol_from_meta(fold)

        ratios.append(ratio)
        vols.append(vol)
        tickers.append(ticker)
        fold_ids.append(gid)

        fold['positive_ratio'] = ratio
        fold['val_vol'] = vol

    if not ratios:
        print("[WARN] No valid folds found to analyze.")
        return

    fold_df = pd.DataFrame({
        'fold_id': fold_ids,
        'ticker': tickers,
        'positive_ratio': ratios,
        'val_vol': vols
    })

    rule_mask = (
    (fold_df['positive_ratio'] >= MIN_POS) & 
    (fold_df['positive_ratio'] <= MAX_POS) &
    (fold_df['val_vol'].fillna(0) >= MIN_VOL))

    df_rule_ok = fold_df[rule_mask].copy()
    df_rule_bad = fold_df[~rule_mask].copy()

    if USE_IQR_OUTLIERS and not df_rule_ok.empty:
        Q1 = df_rule_ok['positive_ratio'].quantile(0.25)
        Q3 = df_rule_ok['positive_ratio'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_mask = (df_rule_ok['positive_ratio'] >= lower_bound) & (df_rule_ok['positive_ratio'] <= upper_bound)
        df_keep = df_rule_ok[iqr_mask].copy()
        df_out_iqr = df_rule_ok[~iqr_mask].copy()
    else:
        df_keep = df_rule_ok.copy()
        df_out_iqr = pd.DataFrame(columns=df_rule_ok.columns)

    restored = []
    med = 0.5
    for tkr, grp in fold_df.groupby('ticker'):
        if (df_keep['ticker'] == tkr).sum() >= MIN_FOLDS_PER_TICKER:
            continue
        pool = df_rule_ok[df_rule_ok['ticker'] == tkr].copy()
        if pool.empty:
            continue
        pool['dist'] = (pool['positive_ratio'] - med).abs()
        need = MIN_FOLDS_PER_TICKER - (df_keep['ticker'] == tkr).sum()
        add_back = pool.sort_values('dist').head(max(0, need))
        if not add_back.empty:
            df_keep = pd.concat([df_keep, add_back.drop(columns=['dist'])], axis=0).drop_duplicates(subset=['fold_id'])
            restored.append((tkr, add_back['fold_id'].tolist()))
    
    df_keep['weight'] = 1.0 - (df_keep['positive_ratio'] - 0.5).abs() * 2.0
    df_keep['weight'] = df_keep['weight'].clip(lower=0.1, upper=1.0)

    kept_ids = set(df_keep['fold_id'].tolist())
    for fold in folds:
        gid = fold.get('global_fold_id')
        if gid in kept_ids:
            w = float(df_keep.loc[df_keep['fold_id'] == gid, 'weight'].iloc[0])
            fold['weight'] = w
            kept.append(fold)
        else:
            if gid in df_rule_bad['fold_id'].values:
                row = df_rule_bad[df_rule_bad['fold_id'] == gid].iloc[0]
                if not (MIN_POS <= row['positive_ratio'] <= MAX_POS):
                    reason = 'ratio_out_of_range'
                elif row['val_vol'] < MIN_VOL:
                    reason = 'low_volatility'
                else:
                    reason = 'other_rule_fail'
            elif gid in df_out_iqr['fold_id'].values:
                reason = 'iqr_outlier'
            else:
                reason = 'unknown'
            dropped.append({**fold, 'drop_reason': reason})

    print(f"Total folds: {len(fold_df)}")
    print(f"Rule-out: {len(df_rule_bad)} (ratio fail: {(df_rule_bad['positive_ratio']<MIN_POS)|(df_rule_bad['positive_ratio']>MAX_POS).sum()}, vol fail: {(df_rule_bad['val_vol']<MIN_VOL).sum()})")
    print(f"IQR-out among rule-OK: {len(df_out_iqr)}")
    if restored:
        print(f"Restored folds per ticker: {restored}")
    print(f"Remaining folds: {len(kept)}")

    if output_json:
        output_path = fold_summary_path.replace(".json", "_cleaned.json")
        with open(output_path, 'w') as f:
            json.dump(kept, f, indent=4)
        drop_path = fold_summary_path.replace(".json", "_dropped.json")
        with open(drop_path, 'w') as f:
            json.dump(dropped, f, indent=4)
        print(f"[INFO] Saved cleaned to: {output_path}")
        print(f"[INFO] Saved dropped to: {drop_path}")

    # PLOTTING 
    plt.figure(figsize=(8, 5))
    plt.hist(fold_df['positive_ratio'], bins=15, edgecolor='black')
    plt.title(f'{model_type.upper()} – Histogram of Positive Label Ratio')
    plt.xlabel('Positive Label Ratio')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    hist_path = os.path.join(fig_dir, f"{model_type}_positive_ratio_hist.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved histogram to: {hist_path}")


    order = fold_df.groupby('ticker')['positive_ratio'].median().sort_values().index.tolist()
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=fold_df, x='ticker', y='positive_ratio', order=order)
    plt.xticks(rotation=90)
    plt.title(f'{model_type.upper()} – Boxplot of Positive Label Ratio per Ticker')
    plt.ylabel('Positive Label Ratio')
    plt.xlabel('Ticker')
    plt.grid(True)
    plt.tight_layout()
    boxplot_path = os.path.join(fig_dir, f"{model_type}_positive_ratio_boxplot.png")
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved boxplot to: {boxplot_path}")

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze fold summary and plot label ratios")
    parser.add_argument("--fold_summary_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=["lstm", "arima"])
    args = parser.parse_args()

    analyze_fold_summary(
        fold_summary_path=args.fold_summary_path,
        model_type=args.model_type
    )
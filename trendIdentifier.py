import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import ttest_1samp
from seaborn import kdeplot
from empiricaldist import Pmf

reits_data = pd.read_csv("reits_data.csv")
reit_groups = reits_data.groupby("Ticker")["Close"]

class CurrentDist:
    def __init__(self):
        self.start = 0
        self.end = None
        self.pmf = Pmf()

    def is_part(self, returns, reject_thresh=0.5):
        """Test wether the returns are likely generated from the current distribution"""

        print(returns)

        if len(self.pmf) == 0:
            self._update_dist(returns)

            return True
        else:
            likelihood = ttest_1samp(a=returns, popmean=self.pmf.max_prob()).pvalue

            if isinstance(likelihood, float) and likelihood >= reject_thresh:
                self._update_dist(returns)

                return True
            elif isinstance(likelihood, float) and likelihood <= reject_thresh:
                return False

    def _update_dist(self, returns):
        for ret in returns:
            if ret in self.pmf.keys():
                self.pmf[ret] += 1
            else:
                self.pmf[ret] = 1

        self.pmf.normalize()

        self.end = self.end + 1 if self.end != None else 1

    def reset_dist(self):
        info = (self.pmf.copy(), self.start, self.end)

        self.pmf = Pmf()
        self.start = self.end

        return info



class EstTrend: 
    def __init__(self, reits_df):
        self.reits_df = reits_df
        self.closings = reits_df.groupby("Ticker")["Close"]
        self.current_dist = CurrentDist()
        self.pmfs = defaultdict(list)
        self.trends = defaultdict(list)

    def gen_log_returns(self, ret_rounding=3):
        log_ret_map = {}
        for closings in self.closings:
            log_returns = round(np.log(closings[1]/closings[1].shift()).dropna(), ret_rounding)
            log_ret_map[closings[0]] = log_returns
        
        self.log_ret_map = log_ret_map
    
    def get_intervals(self, days_steps=7):
        """Gets the intervals for each log return list in the map provided"""
        interval_inds_list = []
        for log_returns in self.log_ret_map.values():
            interval_inds = [i for i in range(len(log_returns)) if i % days_steps == 0]
            interval_inds_list.append(interval_inds)
        
        self.interval_inds_map = {ticker:tuple(log_ret_interval) for log_ret_interval in interval_inds_list for ticker in  self.log_ret_map.keys()}

    def get_ret_in_ret_interval(self):
        ret_int_dict = defaultdict(list)

        for ticker, ret_interval in self.interval_inds_map.items():
            for i in range(len(ret_interval)):
                if i <= len(ret_interval) - 2: 
                    returns_interval = self.log_ret_map[ticker][ret_interval[i]:ret_interval[i+1]]
                    ret_int_dict[ticker].append(tuple(returns_interval))

        self.ret_intervals_map = ret_int_dict

    # def _gen_dist(self, ret_interval):

    #     kde = kdeplot(ret_interval, bw_method="silverman")
        
    #     curve = kde.lines[0]
    #     x, y = curve.get_data()
    #     pmf = Pmf(y, x)
    #     pmf.normalize()  

    #     return pmf


    def check_int_compatibility(self):

        for ticker, ret_intervals in self.ret_intervals_map.items():
            for ret_interval in ret_intervals:

                if self.current_dist.is_part(ret_interval) == False:
                    pmf, start, end = self.current_dist.reset_dist()
                    self.pmfs[ticker].append(pmf)
                    self.trends[ticker].append((start, end))

        return self.pmfs

        



trend = EstTrend(reits_data)
trend.gen_log_returns(3)
trend.get_intervals()
trend.get_ret_in_ret_interval()
print(trend.check_int_compatibility())

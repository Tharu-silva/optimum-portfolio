import streamlit as st
from Stock import Stock
import numpy as np
import math
import pandas as pd

st.set_page_config(page_icon="🏦")


class Portfolio:
    def __init__(self, ticker_1, ticker_2, period="1mo"):
        self.stock_1 = Stock(ticker_1, period)
        self.stock_2 = Stock(ticker_2, period)

    @st.cache
    def get_return(self, w_1, w_2):
        exp_return = w_1 * self.stock_1.mean_return() + w_2 * self.stock_2.mean_return()
        return exp_return

    @st.cache
    def risk(self, w_1, w_2):
        std_1 = self.stock_1.risk()
        std_2 = self.stock_2.risk()

        if w_1 < 0:
            w_sq_1 = -(w_1 ** 2)
            w_sq_2 = w_2 ** 2
        elif w_2 < 0:
            w_sq_1 = w_1 ** 2
            w_sq_2 = -(w_2 ** 2)
        else:
            w_sq_1 = w_1 ** 2
            w_sq_2 = w_2 ** 2

        variance = (w_sq_1 * (std_1 ** 2)) + (w_sq_2 * (std_2 ** 2)) + (2 * self.get_corr() * w_1 * w_2 * std_1 * std_2)
        risk = math.sqrt(variance)
        return risk

    @st.cache
    # Gets the covariance of the two stocks
    def get_cov(self):
        returns1 = self.stock_1.period_returns()
        returns2 = self.stock_2.period_returns()

        # Makes the list of returns the same size
        abs_diff = abs(len(returns1) - len(returns2))
        if len(returns1) > len(returns2):
            returns1 = returns1[abs_diff:]
        else:
            returns2 = returns2[abs_diff:]

        cov_matrix = np.cov(returns1, returns2)

        cov = cov_matrix[0, 1]
        return cov

    @st.cache
    def get_corr(self):
        corr = self.get_cov() / (self.stock_1.risk() * self.stock_2.risk())
        return corr

    @st.cache
    def get_min_risk(self):
        std1 = self.stock_1.risk()
        std2 = self.stock_2.risk()

        numerator = (std2 ** 2) - (self.get_corr() * std1 * std2)
        denominator = (std1 ** 2) + (std2 ** 2) - (2 * self.get_corr() * std1 * std2)

        stock_1_weight = numerator / denominator
        stock_2_weight = 1 - stock_1_weight

        return stock_1_weight, stock_2_weight

    @st.cache
    def get_optimum(self):
        std_1 = self.stock_1.risk()
        std_2 = self.stock_2.risk()
        rtrn_1 = self.stock_1.mean_return()
        rtrn_2 = self.stock_2.mean_return()

        numerator = rtrn_1 * (std_2 ** 2) - rtrn_2 * (self.get_cov())
        denominator = rtrn_1 * (std_2 ** 2) + rtrn_2 * (std_1 ** 2) - (rtrn_1 + rtrn_2) * self.get_cov()

        stock_1_weight = numerator / denominator
        stock_2_weight = 1 - stock_1_weight

        return stock_1_weight, stock_2_weight

    @st.cache
    def plot_risk_vs_return(self):
        weights = []
        curr = 0
        while curr <= 1:
            weights.append(round(curr, 1))
            curr += 0.1

        returns = [self.get_return(x, 1 - x) for x in weights]

        risk = [self.risk(x, 1 - x) for x in weights]

        data = {"Returns (%)": returns,
                "Risk (%)": risk,
                "Stock A Weight": weights}

        df = pd.DataFrame(data)
        return df


st.sidebar.markdown("# Create portfolio")

ticker_a = st.text_input('Please enter stock A')
ticker_b = st.text_input('Please enter stock B')
period = st.selectbox('Select a period', ['1mo', '1wk', '1d'])

a_upper = "_" + "**" + ticker_a.upper() + "**" + "_"
b_upper = "_" + "**" + ticker_b.upper() + "**" + "_"

portfolio = Portfolio(ticker_a, ticker_b, period)

# Sidebar buttons
with st.sidebar:
    a_weight = st.slider('Pick a weight of stock A', min_value=0.0, max_value=1.0)
    b_weight = round(1.0 - a_weight, 3)
    st.write('Stock B weight: ' + str(b_weight))
    get_rtrn = st.checkbox('Get mean return of portfolio')
    risk = st.checkbox('Get risk')
    cov = st.checkbox('Get covariance')
    corr = st.checkbox('Get correlation')
    min_risk = st.checkbox('Get minimum risk')
    optimum = st.checkbox('Get optimum weights')
    plot_risk_vs_return = st.checkbox('Plot risk vs return graph')

try:
    # Event handlers
    if get_rtrn:
        st.markdown("Mean return of portfolio with weights of " +
                 a_upper + ": " + str(a_weight) + " and " + b_upper + ": " + str(b_weight) + " is " +
                 str(round(portfolio.get_return(a_weight, b_weight), 2)) + "%")

    if risk:
        st.markdown("Risk of portfolio with weights of " +
                 a_upper + ": " + str(a_weight) + " and " + b_upper + ": " + str(b_weight) + " is " +
                 str(round(portfolio.risk(a_weight, b_weight), 2)) + "%")

    if cov:
        st.markdown("Covariance of portoflio: " + str(round(portfolio.get_cov(), 2)))

    if corr:
        st.markdown("Correlation coeff. of " + a_upper + " and " + b_upper + ": " + str(round(portfolio.get_corr(), 2)))

    if min_risk:
        a, b = portfolio.get_min_risk()
        st.markdown("Minimum risk portfolio consists of the weights " + a_upper + ": " + str(
            round(a, 2)) + " and " + b_upper + ": " + str(round(b, 2)))

    if optimum:
        a, b = portfolio.get_optimum()
        st.markdown("Optimum portfolio consists of the weights " + a_upper + ": " + str(
            round(a, 2)) + " and " + b_upper + ": " + str(round(b, 2)))

    if plot_risk_vs_return:
        st.markdown("**Graph of risk vs return**")
        st.line_chart(portfolio.plot_risk_vs_return(), x="Risk (%)", y="Returns (%)")
except:
    st.warning('Please enter valid tickers')


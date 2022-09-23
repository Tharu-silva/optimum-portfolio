import streamlit as st
from yahoo_fin.stock_info import *
import numpy as np
from datetime import date


@st.cache
class Stock:
    def __init__(self, ticker, period="1mo"):
        self.ticker = ticker
        self.period = period

    # Gets the adjusted close of a ticker per a certain period
    def prices(self):
        price_data = get_data(self.ticker, start_date="01/01/2015", end_date=date.today(), index_as_date=True,
                              interval=self.period)
        adjclose = price_data.loc[:, 'adjclose']
        return adjclose.tolist()

    # Gets a list of the returns per period in percentage form
    def period_returns(self):
        adjclose_list = self.prices()

        # Loops through the adjusted closes and sums the ubiased daily return divided by the length of the list
        period_returns = []

        for i in range(len(adjclose_list) - 1):
            daily_return = np.log(adjclose_list[i + 1] / adjclose_list[i]) * 100
            if not np.isnan(daily_return):
                period_returns.append(daily_return)

        return period_returns

    # Gets the mean return of the stock per-period
    def mean_return(self):
        period_returns = self.period_returns()

        mean_return = sum(period_returns) / len(period_returns)
        return mean_return

    # Gets the standard deviation of a certain stock
    def risk(self):
        period_returns = self.period_returns()
        return np.std(period_returns, dtype=np.float64)


# User inputs
ticker = st.text_input('Please enter stock ticker')
period = st.selectbox('Select a period', ['1mo', '1wk', '1d'])

stock = Stock(ticker, period)

# Sidebar buttons
with st.sidebar:
    adj_close = st.checkbox('Get period adj. closes', )
    mean_rtr = st.checkbox('Get mean returns')
    risk = st.checkbox('Get risk')

if adj_close:
    try:
        st.line_chart(stock.prices())
    except:
        st.warning("Please enter a valid ticker")

if mean_rtr:
    try:
        st.write(stock.mean_return())
    except:
        st.warning("Please enter a valid ticker")

if risk:
    try:
        st.write(stock.risk())
    except:
        st.warning("Please enter a valid ticker")

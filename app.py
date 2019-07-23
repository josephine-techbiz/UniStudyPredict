#!/usr/bin/env python
# coding: utf-8

# In[2]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.externals import joblib
import plotly.graph_objs as go


# In[3]:


import numpy as np
import dash_daq as daq
import dash_table as dt


# In[4]:


app = dash.Dash()
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})


# In[5]:


training_features = joblib.load("fiturSVM2.pkl")
training_labels = joblib.load("targetSVM2.pkl")


# In[13]:


training_features.head()


# In[ ]:


nem = training_features[['nem']]


# In[6]:


app.layout = html.Div(children=[
    html.H1(children='Prediction using SVM, RT, and RF ', style={'textAlign': 'center'}),

    html.Div(children=[
        html.Label('Masukkan Nilai Ebtanas Murni (NEM) SMA: '),
        dcc.Input(id='nem-sma', placeholder='nem', type='text'),
       # html.Label('Masukkan Kode Jurusan Pilihan: '),
       # dcc.Input(id='kdjur-sma', placeholder='kode jurusan', type='text'),
        html.Div(id='result')
    ], style={'textAlign': 'center'}),
    
    
    
    
    ])
   


# In[7]:


@app.callback(
    Output('result', component_property='children'),
    [Input('nem-sma', component_property='value')
       # Input('kdjur-sma', component_property='value')
    ])
def update_nem_input(nem):
    #return nem
    if nem is not None and nem is not '':
        try:
            ipk = model.predict(nem)[0]
            return 'Dengan NEM: {}  Anda akan memperoleh IPK: {}'.        format(kdjur, ipk, 2)
        except ValueError:
            return 'Tidak dapat memberi IPK'
            


# In[ ]:


if __name__ == '__main__':
    model = joblib.load("modelSVM3.pkl")
    app.run_server()


# In[ ]:





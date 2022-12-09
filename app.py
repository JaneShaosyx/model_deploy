from flask import Flask, jsonify, request
import pickle
import pandas as pd
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

answers = [
    "Strongly agree",
    "Agree",
    "Neither agree nor disagree",
    "Disagree",
    "Strongly disagree",
]

cols = ['MonthlyMaxPrecip', 'MonthlyTotalSnowfall', 'MonthlyMaxSnowfall',
        'MonthlyAvgTemp', 'MonthlyAvgDaylight', 'MonthlyAvgTempDiff',
        'MonthlyAvgPreciptation', 'MonthlyAvgRelativeHumidity',
        'MonthlyAvgWindSpeed', 'MonthlyAvgPeakWindSpeed', 'MonthlyAvgSnowDepth',
        'MonthlyAvgSnowfall', 'MonthlyMaxTempDiff', 'MonthlyMinDailyTempDiff']

predictData = {'MonthlyAvgPreciptation': [0.021, 0.052, 0.118571, 0.163866, 0.196165],
               'MonthlyAvgSnowfall': [0.0, 0.034677, 0.131068, 0.184848, 0.308654],
               'MonthlyTotalSnowfall': [0.5, 1.7, 3.0, 4.9, 7.4],
               'MonthlyAvgDaylight': [618.24, 717.935, 731.270, 827.8333, 930.0548],
               'MonthlyAvgTemp': [9.780000, 13.390000, 17.060000, 23.280000, 25.780000],
               'MonthlyAvgTempDiff': [7.890161, 10.600234, 11.815559, 13.060484, 15.230363],
               'MonthlyAvgRelativeHumidity': [48.875000, 59.578902, 66.933333, 74.016129, 77.643805],
               'MonthlyAvgWindSpeed': [5.055667, 7.082500, 8.466129, 9.668548, 10.903379],
               'MonthlyMaxPrecip': [0.46, 0.72, 0.97, 1.63, 2.27],
               'MonthlyMaxSnowfall': [0.0, 0.2, 1.7, 2.5, 3.9],
               'MonthlyAvgSnowDepth': [0.0, 0.016807, 0.200000, 0.412844, 1.147826],
               'MonthlyMaxTempDiff': [15.560000, 18.330000, 20.835000, 23.890000, 26.110000],
               'MonthlyMinDailyTempDiff': [1.670000, 2.220000, 2.780000, 3.340000, 4.450000],
               'MonthlyAvgPeakWindSpeed': [20.303279, 23.868175, 26.396129, 29.169355, 31.796460]}


@app.route("/", methods=['POST', 'GET'])
def index():
    if (request.method == 'POST'):
        data = request.get_json()
        month = data['month']
        df = pd.DataFrame(data, index=[0]).drop(['month'], axis=1)
        for c in cols:
            idx = answers.index(data[c])
            df[c] = predictData[c][idx]
        df = df[cols]
        lin_reg = pickle.load(open(f"./models/month_{month}", 'rb'))
        cluster = lin_reg.predict(df).tolist()[0]
        return jsonify(cluster)
        return jsonify({})
    else:
        return jsonify({"about": "Hello World"})


if __name__ == '__main__':
    app.run(port=8888, host='localhost', debug=True)

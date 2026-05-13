from flask import Blueprint, render_template, request
import time  
from .predictor import Predictor

bp = Blueprint('main', __name__)
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = Predictor()
    return predictor

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    
    time.sleep(1.5)
    
    res = get_predictor().predict(data)
    
    return render_template('result.html', result=res, input=data)
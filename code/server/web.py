from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import train_blur
import train_blur_ellipse
import train_mosaic
import train_mosaic_ellipse
import os

# 현재 파일의 위치를 기준으로 templates 폴더 경로 설정
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
app = Flask(__name__, template_folder=template_dir)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/input', methods=['POST'])
def inputs():
    image_path=os.path.abspath(os.path.join(os.path.dirname(__file__),'static/upload'))
    file = request.files['file']
    filename= secure_filename(file.filename)
    file.save(image_path+'/'+filename)
    return render_template('input.html',link_folder_data='/static/upload/',link_file_data=filename)

@app.route('/input/preview', methods=['POST'])
def preview():
    if request.args.get('data') == 'blur':
        # 다른 Python 파일 실행
        train_blur.tarin_blur(request.args.get('filename'))
    elif request.args.get('data') == 'blur_ellipse':
        # 다른 Python 파일 실행
        train_blur_ellipse.train_blur_ellipse(request.args.get('filename'))
    elif request.args.get('data') == 'mosaic':
        # 다른 Python 파일 실행
        train_mosaic.mosaic(request.args.get('filename'))
    elif request.args.get('data') == 'mosaic_ellipse':
        # 다른 Python 파일 실행
        train_mosaic_ellipse.mosaic_ellipse(request.args.get('filename'))
    return render_template('preview.html', data=request.args.get('data'),file_name=request.args.get('filename'))

@app.route('/input/result', methods=['POST','GET'])
def input_result():
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static/download/'+request.args.get('data')+'_'+request.args.get('file_name')))
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)

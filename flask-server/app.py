import re
from datetime import datetime
import time
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
#from config import OPEN_API_KEY, PINECONE_KEY, SERP_API_KEY
from demoo import initialize, process_and_store_documents, get_my_agent
import demoo
import njo61u04
from werkzeug.utils import secure_filename
import os
import tempfile
import json
from njo61u04 import get_my_agent_1
from ttsnstt import speech_to_text, text_to_speech

from sql import get_data_from_table, deleteid, process_and_store_documents

app = Flask(__name__)
CORS(app)
#CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})

# Access the config variables
#app.config['OPEN_API_KEY'] = OPEN_API_KEY
#app.config['SERP_API_KEY'] = SERP_API_KEY
#app.config['PINECONE_KEY'] = PINECONE_KEY





# 測試文本轉語音
#test_text = "Hello, this is a test for text to speech."
#output_audio_file = "output/audio/path/output.wav"  # 替換為要保存語音文件的路徑
#text_to_speech(api_key, region, test_text, output_audio_file)
#print("TTS Result: Audio saved to", output_audio_file)

#https://code.visualstudio.com/docs/python/tutorial-flask#_use-a-template-to-render-a-page

embeddings = initialize()

#source_text = "無"
#source_doc = "無"
#print(f"\n --- \n Agent prompt:\n {my_agent.agent.llm_chain.prompt}\n")
#print(f"Agent Output Parser: {my_agent.agent.llm_chain.prompt.output_parser}\n---\n")



@app.route('/get_answer', methods=['POST'])
@cross_origin()
def process_input():
    data = request.get_json()
    print(f'server receive data from User input: {data}')
    user_input = data.get('inputText') #等同於 data['inputText']
    text_length = data.get('textLength')

    my_agent= get_my_agent(text_length)
    #similarity_QA = ['來源1', '來源2','來源3']
    source = [" "]
    Ai_response = " "
    flowchart_url = " "
    QA_onlyQ = " "
    print("ssss")
    if "流程圖" in user_input :
        try:
            print("user:"+ user_input)
            flowchart_url = demoo.flowchart(user_input)
            Ai_response = "流程圖："+ flowchart_url
            print("flowchart_url" + flowchart_url)
        except Exception as e:
            print(e)

    elif "(client_Q)" in user_input :
        try:
            print("user:"+ user_input)
            QA_onlyQ = demoo.QQQ(user_input)
            print("QA_onlyQ" + QA_onlyQ)
            Ai_response = "QA: " + QA_onlyQ
        except Exception as e:
            print(e)

    else:
        try:
            Ai_response = my_agent.run(user_input)
        except Exception as e:
            Ai_response = str(e)
            print(f'\nThe error message is here: {e}')
            if Ai_response.startswith("Could not parse LLM output: `"):
                Ai_response = Ai_response.removeprefix("Could not parse LLM output: `").removesuffix("`")
            else:
                Ai_response = my_agent.run("I want to know" + user_input)

        # Initialize the variables to "no message" by default

        try:
            source_text = demoo.similarity_QA
            source_doc = demoo.similarity_QA_source
            source = [i + "\n\n" + json.dumps(j, ensure_ascii=False) for i, j in zip(source_text, source_doc)]
            print(source)
        except:
            pass
        
        return jsonify({'response': Ai_response, 'source': source})
    



#正則過濾網址
def replace_urls_with_text(text, replacement="網址"):
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, replacement, text)




@app.route('/get_audio/<filename>', methods=['GET'])
def get_audio(filename):
    """向前端發送音頻文件"""
    return send_file(filename, mimetype='audio/wav', as_attachment=True)


@app.route('/receive_audio', methods=['POST'])
@cross_origin()
def receive_audio():
    my_agent_1= get_my_agent_1()
    audio_file = request.files['audio']
    
    # 將檔案保存到臨時檔案中
    temp_audio_path = "temp_audio.mp3"
    audio_file.save(temp_audio_path)

    # 呼叫 Azure OpenAI 語音轉文本
    api_key = "api_key"
    resource_name = "resource_name"
    deployment_id = "deployment_id"
    api_version = "api_version"
    text_result = speech_to_text(api_key, resource_name, deployment_id, api_version, temp_audio_path)

    # 刪除臨時檔案
    os.remove(temp_audio_path)

    try:
        # 解析 JSON 響應
        response_json = json.loads(text_result)
        extracted_text = response_json.get("text", "")
    except json.JSONDecodeError:
        # 處理 JSON 解析錯誤
        return jsonify({'error': 'Failed to decode response from Azure OpenAI'})
    

    print("Extracted Text:", extracted_text)
    imspeaktext = extracted_text+'try Voice_message.'

    print("傳送:", imspeaktext)
    try:
        Ai_response = my_agent_1.run(imspeaktext)
    except Exception as e:
        Ai_response = str(e)
        print(f'\nThe error message is here: {e}')
        if Ai_response.startswith("Could not parse LLM output: `"):
            Ai_response = Ai_response.removeprefix("Could not parse LLM output: `").removesuffix("`")
        else:
            Ai_response = my_agent_1.run("I want to know" + imspeaktext)

    try:
        source_text = njo61u04.similarity_QA
        source_doc = njo61u04.similarity_QA_source
        source = [i + "\n\n" + json.dumps(j, ensure_ascii=False) for i, j in zip(source_text, source_doc)]
        print(source)
    except:
        pass


    # 在這裡過濾網址
    Ai_response_clean = replace_urls_with_text(Ai_response)

    # 調用 Text-to-Speech
    api_key = "api_key"
    region = "region"
    audio_content = text_to_speech(api_key, region, Ai_response_clean)


    if isinstance(audio_content, bytes):
        # 将音频数据保存为临时文件
        temp_audio_path = "temp_audio.wav"
        with open(temp_audio_path, "wb") as audio_file:
            audio_file.write(audio_content)
        # 在原有的JSON响应中加入音频文件路径
        return jsonify({'response': Ai_response, 'source': source, 'extracted_text':extracted_text, 'audio_path': temp_audio_path})
    else:
        return jsonify({'response': Ai_response, 'source': source, 'extracted_text':extracted_text, 'audio_path': None})


    #return jsonify({'text': extracted_text})




TABLE_NAME_MAPPING = {
        "Table1": "index",
        "Table2": "flowchart",
        "Table3": "indexqa",
        "Table4": "indexsynonym",
        "Table5": "q"
}


@app.route('/api/data/<table_name>', methods=['GET'])
def get_data(table_name):
    
    
    db_table_name = TABLE_NAME_MAPPING.get(table_name)
    if not db_table_name:
        print(f"Invalid table name: {table_name}")
        return jsonify({"error": "Invalid table name"}), 400

    #condition = request.args.get('condition', '1=1')
    data = get_data_from_table(db_table_name)
    return jsonify(data)


@app.route('/api/delete/<table_name>/<int:item_id>', methods=['DELETE'])
def delete_item(table_name, item_id):

    db_table_name = TABLE_NAME_MAPPING.get(table_name)
    try:
        deleteid(db_table_name, item_id)
        return jsonify({"message": "Item deleted successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/api/upload_doc', methods=['POST'])
def upload_documents():
    responses = []
     # 從請求中獲取 selectedTable 數據
    selected_table = request.form['selectedTable']
    db_table_name = TABLE_NAME_MAPPING.get(selected_table, None)


    print(f'\nfile objects that I receive in this upload:{request.files}')
    print(f'Selected table: {db_table_name}')  # 打印當前選中的表格名稱

    if not db_table_name:
        return jsonify({'message': 'Invalid table selection.'}), 400
    
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request.'}), 400

    files = request.files.getlist('file')  # Get the list of files
    if files[0].filename.endswith('/'):
        print('haha')
        folder_path = files[0].filename
        file_names = os.listdir(folder_path)
        files = [open(os.path.join(folder_path, file_name), 'rb') for file_name in file_names]

    for file in files:
        print('\nthis file:', file)
        if file.filename == '':
            responses.append({'message': '''There's a bad file.''', 'code': 400})
        
        # Save the file to a temporary location with original extension
        extension = os.path.splitext(file.filename)[-1].lower()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
        file.save(temp_file.name)
        print("Temporary file path:", temp_file.name)
        
        try:
            process_and_store_documents([temp_file.name], db_table_name, file.filename)
            responses.append({
                'message': f'File uploaded and processed successfully to Temporary space {temp_file.name}', 'code': 200
            })
        except Exception as e:
            app.logger.debug('Debug message')
            responses.append({'message': f'Error processing file: {str(e)}', 'code': 500})
        finally:
            try:
                # Delete the temporary file
                os.remove(temp_file.name)
            except PermissionError:
                print(f"Unable to delete {temp_file.name}. It might be in use by another process.")

    
    responses.append('end message:file upload successfully finished')
    return jsonify({'responses': responses}), 200


#. .venv/bin/activate  
#python -m flask run
if __name__ == '__main__': 
    app.run(debug=True, port=80)
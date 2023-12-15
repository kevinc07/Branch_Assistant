import React, { useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faL, faStar as solidStar } from '@fortawesome/free-solid-svg-icons';
import { faStar as regularStar } from '@fortawesome/free-regular-svg-icons';
import { ReactMic } from 'react-mic';
import axios from 'axios';
import Linkify from 'react-linkify';
import fub_pic from './assets/fub.jpg';
import role_pic from './assets/role.png';
import loading_pic from './assets/money.png';

function App() {

  // 開啟聊天視窗
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [view, setView] = useState('Chating');


  //當有可用來源時，才調用
	const [sourceAvailable, setSourceAvailable] = useState(false);
	const [sourceItems, setSourceItems] = useState([]);
	const [selectedIndex, setSelectedIndex] = useState(null);
  const [currentSourceItems, setCurrentSourceItems] = useState([]);



  // 搜尋關鍵字
  const [searchString, setSearchString] = useState('');

  const [isSourceOpen, setSourceOpen] = useState(false);

  const linkDecorator = (href, text, key) => (
    <a
    href={href}
    key={key}
    target="_blank"
    style={{ color: 'blue' }}>{text}</a>
  );
  

  // 收藏對話
  const [savedMessages, setSavedMessages] = useState([]);

  // 編輯筆記本
  const [editingItem, setEditingItem] = useState(null);
  const [editingIndex, setEditingIndex] = useState(null);


  const handleEditOrSave = (index) => {
    if (editingIndex === index) {
      // 如果已經在編輯這個項目，那麼保存更改
      notebookContent[index] = editingItem;
      setEditingIndex(null);
    } else {
      // 否則，開始編輯這個項目
      const itemToEdit = { ...notebookContent[index] };
      itemToEdit.date = new Date().toLocaleString();
      setEditingItem(itemToEdit);
      setEditingIndex(index);
    }
  }

  const [hasPlayed, setHasPlayed] = useState(false);  // 添加播放状态

  //音檔上傳
  const uploadAudioFile = async (audioFile) => {
    const formData = new FormData();
    formData.append('audio', audioFile);

    try {
      const response = await fetch('http://20.243.26.237:80/receive_audio', { // 替換為您的後端 API 端點
        method: 'POST',
        body: formData,
      });
  
      if (response.ok) {
        console.log('Audio file uploaded successfully');

        const responseData = await response.json(); // 解析 JSON 響應
        const extractedText = responseData.text; // 從響應中提取文本
        setUserInput(extractedText); // 更新 input 欄位的狀態
      } else {
        console.error('Failed to upload audio file');
      }
  
    } catch (error) {
      console.error('Error uploading audio file', error);
    }
  }

  

  //語音錄製
  // const [isRecording, setIsRecording] = useState(false);
  // const [audioData, setAudioData] = useState(null);

  // const startRecording = () => {
  //   setIsRecording(true);
  // };

  // const stopRecording = () => {
  //   setIsRecording(false);
  // };

  // const onData = (recordedData) => {
  //   // 可以用於更新音頻可視化
  // };

  // const onStop = (recordedBlob) => {
  //   setAudioData(recordedBlob);
  //   // 在這裡可以選擇直接發送錄音數據到後端
  //   sendAudioToBackend(recordedBlob.blob);
  // };

  // const sendAudioToBackend = async (audioBlob) => {
  //   const formData = new FormData();
  //   formData.append('audio', audioBlob);

  //   try {
  //     const response = await fetch('/receive_audio', { // 替換為您的後端 API 端點
  //       method: 'POST',
  //       body: formData,
  //     });

  //     if (response.ok) {
  //       console.log('Audio sent successfully');
  //       // 處理後端返回的響應
  //     } else {
  //       console.error('Failed to send audio');
  //     }

  //   }catch (error) {
  //     console.error('Error sending audio', error);
  //   }
  // }




  // 儲存對話內容的狀態
  const [notebookContent, setNotebookContent] = useState([]);
  const [dialogueContent, setCurrentDialogueContent] = useState([]);

  const [otherMenuVisible, setotherMenuVisible] = useState(false);
  const toggleMenuVisibility = () => {
    setotherMenuVisible(!otherMenuVisible);
  }


  // 邦妮QA及作業細則點選
  const [menuVisible, setMenuVisible] = useState(false);
  const [selectedButtons, setSelectedButtons] = useState({
    bonnieQA: true,
    assignmentDetails: true,
    clientsQ: false,
  })

  const handlebtnClick = (type) => {
    setSelectedButtons({ ...selectedButtons, [type]: !selectedButtons[type] });
    setMenuVisible(false); // 選擇後關閉選單
  }


  // 定義使用者點擊AI對話氣泡
  const [userInput, setUserInput] = useState('');
  const [messages, setMessages] = useState([{ HumanInput: '', AiResponse: '很高興為您服務。' }]); // 存儲本次聊天室對話紀錄
  const [audioPath, setAudioPath] = useState('');
  const [loading, setLoading] = useState(false);

  const handleInputChange = (e) => {
    setUserInput(e.target.value);
  };

  // 儲存對話記錄的功能
  const handleSaveMessage = (message) => {
    setSavedMessages((prevSavedMessages) => [...prevSavedMessages, message]);
  }

  const handleClick = (buttonName, event) => {
    event.stopPropagation();

    setSelectedButtons((prevState) => {
      // 如果該按鈕已被選中，我們檢查其他按鈕是否未被選中。
      if (prevState[buttonName]) {
        const otherButtons = Object.keys(prevState).filter((btn) => btn !== buttonName);
        if (!prevState[otherButtons[0]] && !prevState[otherButtons[1]]) {
          // 如果其他兩個按鈕都未被選中，直接返回當前狀態，不作更改
          return prevState;
        }
      }
      return {
        ...prevState,
        [buttonName]: !prevState[buttonName],
      }
    })
  }

  // 編輯筆記本
  const handleEdit = (index) => {
    console.log('Editing index:', index);
    const itemToEdit = { ...notebookContent[index] };

    // 更新日期
    itemToEdit.date = new Date().toLocaleDateString();
    console.log('Item to edit:', itemToEdit);
    setEditingItem(itemToEdit);
    setEditingIndex(index);
  }


  const updateEditingItem = (propertyName, value) => {
    setEditingItem((prevItem) => ({
      ...prevItem,
      [propertyName]: value,
    }))
  }


  // 高亮搜尋字串
  const highlightSearchString = (content) => {
    if (!searchString) return content; // 如果搜尋字串不存在，返回原始文本

    const regex = new RegExp(`(${searchString})`, 'gi');
    return content.replace(regex, '<span style="background-color: yellow;">$1</span>');
  }

  // 刪除收藏
  const handleDelete = (index) => {
    const isConfirmed = window.confirm('確定刪除？');
    if (isConfirmed) {
      setNotebookContent((prevContent) => prevContent.filter((_, i) => i !== index));
    }
  }

  // 定義檢查個資
  const personalDataPatterns = [
    /(?:\d{4}[ -]){3}\d{3,4}/, // 信用卡號碼
    /[A-Z]{2}\d{8}/, // 居住證號
    /[\u4E00-\u9FA5]+(?:路|街|大道)(?:\d+段)?(?:\d+巷)?(?:\d+弄)?\d+號(?:\d+樓)?/, // 地址
    /[\w-]+@[\w-]+(?:\.[\w-]+)+/, // 電子郵件
    /09\d{8}|\d{2,4}-\d{6,8}|0\d{7,11}/, // 電話號碼
    /[A-Z]\d{9}/, // 身份證號碼
  ];

  const handleSendMessage = async (e) => {
    e.preventDefault();
    setLoading(true);

    if(userInput=="") {
      setLoading(false);
      return;
    }

    // 檢查 userInput 是否含有可能的個人資料
    for (const pattern of personalDataPatterns) {
      if (pattern.test(userInput)) {
        alert('請勿輸入個人資料!');
        setLoading(false);
        return;
      }
    }

    if (!selectedButtons.bonnieQA && !selectedButtons.assignmentDetails && !selectedButtons.clientsQ) {
      alert('邦妮QA和作業細則不能都不點選哦!!');
      setLoading(false); // 可能需要重置加載狀態
      return;
    }

    const flow = '流程圖';
    let text = userInput;

    if (userInput.includes(flow)) {
      text = userInput;
    } else if (selectedButtons.clientsQ) {
      text += '(client_Q)';
    } else if (selectedButtons.bonnieQA && !selectedButtons.assignmentDetails) {
      text += ' firstly, try QAsearch_First if there are not engough information you can try QAsearch_second';
    } else if (!selectedButtons.bonnieQA && selectedButtons.assignmentDetails) {
      text += ' firstly, try DOCsearch_only if there are not engough information you can try DOCsearch_synonym';
    } else {
      text = userInput;
    }

    setUserInput(''); // 清空輸入框
    axios.post('http://20.243.26.237:80/get_answer', {
      inputText: text,
    })
    .then(function (response){
      var status = response.status;
      const resp = status === 200 ? response.data.response : '暫時無法解析您的問題';
      const source = response.data.source;
      //const myurl = response.data.flowchart_url;
      //const myQA = response.data.QA_onlyQ;
      
      if (response.data.source && response.data.source.length > 0) {
        // 如果有來源數據，那麼更新 sourceItems 和 sourceAvailable 的狀態
				setSourceItems(response.data.source);
				setSourceAvailable(true);
      } else {
        setSourceItems([]);
				setSourceAvailable(false);
      }

      const audioPath = response.data.audio_path; // 获取音频文件路径
      //console.log('flaskRes:', resp)
      const newMessage = {
        HumanInput: userInput,
				AiResponse: resp,
				sourceAvailable: source.length >= 3,
    		sourceItems: source,
				date: new Date().toLocaleString() // 儲存當前日期和時間
      }

      handleSaveMessage(newMessage); 
      setMessages((prevMessages) => [...prevMessages, newMessage]);
      setCurrentDialogueContent([...dialogueContent, newMessage]);

      // 重置hasPlayed状态以便下一次播放
      if (audioPath) setHasPlayed(false);
    })
    .catch(function (err){
			alert(`Error: ${err}`);
		})
    .finally(() => {
			setLoading(false)
		});
  }

  // useEffect(() => {
  //   if (audioPath && !hasPlayed) {
  //     fetchAudioAndPlay(`http://20.243.26.237:80/get_audio/${audioPath}`);
  //   }
  // }, [dialogueContent]); // 依赖于dialogueContent的变化


  // 获取并播放音频文件的函数
  const fetchAudioAndPlay = (audioUrl) => {
    if (!hasPlayed) { 
      fetch(audioUrl)
      .then(response => response.blob())
      .then(blob => {
        const audioBlobUrl = URL.createObjectURL(blob);
        const audio = new Audio(audioBlobUrl);
        audio.play();
        setHasPlayed(true);  // 更新播放狀態
      })
      .catch(err => console.error('Error playing audio:', err));
    }
  };



  //文本上傳資料庫
  const [selectedTable, setSelectedTable] = useState(null);
  const [tableData, setTableData] = useState([]);
  const [fullViewId, setFullViewId] = useState(null);


  // 這個函數可以被重複使用來獲取和更新表格數據
  const fetchDataForSelectedTable = () => {
    if (selectedTable) {
      axios.get(`/api/data/${selectedTable}`)
      .then((response) => {
        const formattedData = response.data.map(item => {
          const parsedItem = JSON.parse(item[1]);
          const name = parsedItem.source ? parsedItem.source.split('\\').pop() : 'Unknown';
          return {
            id: item[0],
            name: name,
            value: item[2]
          }
        })
        setTableData(formattedData);
      })
      .catch((error) => {
        console.error('Error fetching data:', error);
      });
    }
  }

 useEffect(() => {
    // 在這個 effect 中擷取資料
    fetchDataForSelectedTable();
  }, [selectedTable])

  
  const handleTableSelect = (table) => {
    setSelectedTable(table);
  };

  {/*const handleClearSelection = () => {
    setSelectedTable(null);
  };*/}

  const handleFullViewToggle = (id) => {
    if (fullViewId === id) {
      setFullViewId(null); // 如果已經是完整視圖，則切換回摘要視圖
    } else {
      setFullViewId(id); // 否則切換到完整視圖
    }
  }

  const handleDeleteTable = (id) => {
    if (!selectedTable || !id) {
      console.error('Invalid table or ID.');
      return;
    }

    // 確認刪除
    const isConfirmed = window.confirm('您確定要刪除此項目嗎？');
    if (!isConfirmed) {
      return;
    }


    // 這裡可以發送請求到後端刪除該項目，然後更新前端的狀態
    axios.delete(`/api/delete/${selectedTable}/${id}`)
      .then(() => {
        // 從tableData中移除該項目
        const updatedData = tableData.filter(item => item.id !== id);
        setTableData(updatedData);
      })
      .catch((error) => {
        console.error('Error deleting item:', error);
      })
  }


  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
        setSelectedFile(file);
    }
  };

  const handleUploadClick = () => {
    if (selectedFile) {
      
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('selectedTable', selectedTable);
      
      // 發送文件和 selectedTable 到後端      
      fetch('/api/upload_doc', {
        method: 'POST',
        body: formData,
      })
      .then((response) => response.json())
      .then((data) => {
        console.log('從後端收到的數據:', data);
        fetchDataForSelectedTable(); // 添加這個調用來獲取最新資料
      })
      .catch((error) => {
        console.error('上傳文件時出錯:', error);
      })
    }
  }




  return (
    <>
    {/* chat按鈕 */}
    <button
    className="fixed bottom-0 left-0 m-6 bg-gradient-blue text-white rounded-full p-4 shadow-lg btn-animate transition-custom"
    onClick={() => setIsChatOpen((prev) => !prev)}>
      分行助手
    </button>

    <div className="flex h-screen">
      {/* 左側資料來源面板 */}
      <div className="flex-1">
        <iframe
        id="leftIframe" 
        style={{ width: '50%', height: '100%', border: 'none' }}/>

        {/* 新增一個div作為左側資料來源面板 */}
        <div className={`absolute top-10 left-10 p-6 h-[80%] w-[45%] bg-finalbutter transition-transform transform border border-gray-300 shadow-lg rounded-lg ${isSourceOpen ? 'translate-x-0' : '-translate-x-full'}`}>
          <div className="overflow-y-auto mb-4" style={{ maxHeight: 'calc(100% - 40px)', fontSize:'18px' }}>
            {isSourceOpen && currentSourceItems[selectedIndex]}
          </div>
          <button className="px-4 py-2 bg-finalbluu text-white rounded hover:bg-blue-600 focus:outline-none focus:bg-blue-700" onClick={() => setSourceOpen(false)}>
            close
          </button>
        </div>
      </div>

      {/* chat視窗 */}
      <div className="flex-1">
        {isChatOpen && (
          <div className="fixed top-4 right-4 h-[95%] w-[50%] flex flex-col p-4 shadow-lg rounded-lg border-2 border-white bg-finalyellow">
            
            {/* Close button */}
            <button
            className="self-end text-finalred font-bold rounded-full p-3 mb-2 btn-animate transition-custom"
            onClick={() => setIsChatOpen(false)}
            >
              關閉
            </button>
            
            {/* 聊天框 */}
            {view === 'Chating' ? (
              <section className="flex flex-col items-center h-[90%] bg-finalbutter border-2 border-finalred shadow-inner rounded-lg ">
                <div className="mt-2 bg-gray-830 flex sticky top-1 h-[50px] w-[98%] mb-2 border-b-2 border-finalred z-index-10">
                  <div className="text-3xl text-start font-semibold text-finalred flex-grow">
                    分行助手
                  </div>
                  <div
                  className="text-end font-semibold text-finalred cursor-pointer hover:underline"
                  onClick={() => { setView('Notebook'); }}>
                    📓個人筆記本
                  </div>
                  <div
                  className="text-end font-semibold text-finalred cursor-pointer hover:underline"
                  onClick={() => { setView('Settings'); }}>
                    ⚙️資料管理
                  </div>
                  <input type="file" onChange={(e) => uploadAudioFile(e.target.files[0])} accept=".mp3"/>
                </div>
                
                <div className="h-[80%] w-[95%] overflow-y-auto">
                  <div>
                    {messages.map((message, index) => (
                      <div>
                        <div
                        key={index}
                        className="flex-col gap-2 mb-3">
                          {message.HumanInput && (
                            <div className="flex items-center justify-end gap-2">
                              {' '}
                              {/* 使用者對話框 */}
                              <div className="bg-finalyellow text-grey-500 rounded-lg p-2 max-w-[60%] break-words">
                                {message.HumanInput}
                              </div>
                              <img
                              alt=""
                              className="mr-2 h-10 w-10 rounded-full object-cover"
                              src={role_pic}>
                              </img>
                            </div>
                          )}
                          <div className="text-gray-400 text-right">
                            {message.date}
                          </div>
                          {' '}
                          {/* 日期位置，使用text-right來對齊到右邊 */}
                        </div>
                      {/* 使用者對話框 */}


                      {/* 機器人對話框 */}
                      <div className="relative gap-2 mb-4">
                        <div className="flex items-start justify-start">
                          <img 
                          alt=""
                          className="h-10 w-10 rounded-full object-cover"
                          src={fub_pic}>
                          </img>
                          {/* 新增一個flex容器 */}
                          <div className="flex items-center justify-start max-w-[90%]">
                            <div
                              style={{whiteSpace: 'pre-wrap', wordWrap: 'break-word', overflowWrap: 'break-word', wordBreak: 'break-all'}}
                              className="ml-2 bg-white text-black rounded-lg p-2 border border-finalred">
                                <Linkify componentDecorator={linkDecorator}>
                                  {message.AiResponse}
                                </Linkify>
                                {message.sourceAvailable && (
                                  <>
                                  <hr />
                                  <div className="mt-3 flex flex-row gap-2">
                                    {message.sourceItems.map((source, index) => (
                                      <button
                                      key={index}
                                      className="bg-gray-200 hover:bg-blue-300 text-black p-2 rounded btn-animate transition-custom"
                                      onClick={() => {
                                        setSelectedIndex(index);
                                        setCurrentSourceItems(message.sourceItems);
                                        setSourceOpen(true);
                                      }}>
                                        資料來源
                                      </button>
                                    ))}
                                  </div>
                                  </>
                                )}
                            </div>
                            {/* 收藏按鈕 */}
                            <button
                            className="mr-5 p-2 rounded btn-animate transition-custom"
                            onClick={() => {
                              console.log('收藏按鈕被點擊');
                              const dialog = {
                                question: message.HumanInput,
                                answer: message.AiResponse,
                                date: message.date
                              }

                              //檢查是否存在相同問題和答案
                              if (
                                notebookContent.some(
                                  (content) => content.question === dialog.question && content.answer === dialog.myQA
                                )
                              ) {
                                //如果存在相同問題和答案，則刪除
                                setNotebookContent((prevContent) =>
                                prevContent.filter(
                                  (content) => content.question !== dialog.question || content.answer !== dialog.answer
                                ))
                              } else {
                              //否則將對話添加到筆記本中
                              setNotebookContent((prevContent) => [...prevContent, dialog])
                              }
                            }}>
                              {notebookContent.some(
                                (content) =>
                                content.question === message.HumanInput && content.answer === message.AiResponse
                              ) ? (
                                <FontAwesomeIcon 
                                icon={solidStar} 
                                color='gold'>
                                </FontAwesomeIcon>
                              ) : (
                                <FontAwesomeIcon
                                icon={regularStar}
                                color="darkgray">
                                </FontAwesomeIcon>
                              )}
                            </button>
                          </div>
                        </div>
                        <div className="text-gray-400 text-left">
                          {message.date}
                        </div>
                      </div>
                      {/* 機器人對話框 */}
                    </div>
                  ))}

                  {loading && (
                    <div className="sticky top-0 left-0 h-full flex items-center justify-center">
                      <div className="flex items-center justify-center bg-white border p-4 rounded-lg w-64">
                        <div className="mr-4" role="status">
                          <svg 
                          aria-hidden="true"
                          className="w-20 h-20 mr-2 text-gray-200 animate-spin dark:text-gray-500 fill-blue-600" 
                          viewBox="0 0 100 101" 
                          xmlns="http://www.w3.org/2000/svg">
                            <image xlinkHref={loading_pic} height="100" width="100" />
                          </svg>
                        </div>
                        <div>
                          <p>您的資料正在圖書館找尋答案，請稍待片刻，我們將精心為您服務.....</p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
              <div className="mt-4 mx-auto z-0 flex h-[8%] w-[100%] flex-row items-center rounded-xl px-4">
                <form
                className="px-6 mx-auto flex w-full flex-col" // 讓元件垂直排列
                onSubmit={handleSendMessage}>
                  <div className="w-full flex items-center mb-2">
                   
                    {/* <div
                    className='relative cursor-pointer'
                    onClick={toggleMenuVisibility}>
                      <svg 
                        xmlns="http://www.w3.org/2000/svg" 
                        fill="none" 
                        viewBox="0 0 24 24" 
                        strokeWidth={1.5} 
                        stroke="#631f16" 
                        className="w-10 h-10 btn-animate transition-custom">
                          <title>點擊來源查詢</title>
                          <path 
                          strokeLinecap="round" 
                          strokeLinejoin="round" 
                          d="M15.75 15.75l-2.489-2.489m0 0a3.375 3.375 0 10-4.773-4.773 3.375 3.375 0 004.774 4.774zM21 12a9 9 0 11-18 0 9 9 0 0118 0z">
                          </path>
                      </svg>
                      {otherMenuVisible && (
                        <div className="absolute bg-white bottom-full right-0 shadow rounded p-4 flex flex-col z-100 border-2 border-finalred">
                          <h2 style={{ fontSize: '1.5rem' }}>選擇您的答案長度</h2>
                          <div className='w-32'>
                            <input type="radio" id="short" name="responseLength" value="short"/>
                            <label htmlFor="short">短文本</label>
                          </div>
                          <div className='w-32'>
                              <input type="radio" id="medium" name="responseLength" value="medium" defaultChecked/>
                              <label htmlFor="medium">中等文本</label>
                          </div>
                          <div className='w-32'>
                              <input type="radio" id="long" name="responseLength" value="long"/>
                              <label htmlFor="long">長文本</label>
                          </div>
                        </div>
                        )}
                    </div> */}
                    <div 
                    className="relative cursor-pointer" 
                    onClick={(event) => event.stopPropagation() || setMenuVisible(!menuVisible)}>
                      <svg 
                      xmlns="http://www.w3.org/2000/svg" 
                      fill="none" 
                      viewBox="0 0 24 24" 
                      strokeWidth={1.5} 
                      stroke="#631f16" 
                      className="w-10 h-10 btn-animate transition-custom">
                        <title>點擊來源查詢</title>
                        <path 
                        strokeLinecap="round" 
                        strokeLinejoin="round" 
                        d="M15.75 15.75l-2.489-2.489m0 0a3.375 3.375 0 10-4.773-4.773 3.375 3.375 0 004.774 4.774zM21 12a9 9 0 11-18 0 9 9 0 0118 0z">
                        </path>
                      </svg>
                      {menuVisible && (
                        <div className="absolute bg-white bottom-full right-0 shadow rounded p-4 flex flex-col z-100 border-2 border-finalred">
                          <button
                          type="button"
                          onClick={(event) => handleClick('bonnieQA', event)}
                          className={`px-4 py-2 w-32 font-semibold rounded btn-animate transition-custom text-white ${selectedButtons.bonnieQA ? 'bg-purple-300' : 'bg-gray-400'}`}
                          title="只在邦妮QA內找資料">
                            邦妮QA
                          </button>
                          <button
                          type="button"
                          onClick={(event) => handleClick('assignmentDetails', event)}
                          className={`px-4 mt-2 py-2 font-semibold w-32 rounded btn-animate transition-custom text-white ${selectedButtons.assignmentDetails ? 'bg-pink-300' : 'bg-gray-400'}`}
                          title="只在作業細則內找資料">
                            作業細則
                          </button>
                          <button
                          type="button"
                          onClick={(event) => handleClick('clientsQ', event)}
                          className={`px-4 mt-2 py-2 font-semibold w-32 rounded btn-animate transition-custom text-white ${selectedButtons.clientsQ ? 'bg-blue-300' : 'bg-gray-400'}`}
                          title="3秒內給你答案">
                            快速查找
                          </button>
                        </div>
                      )}
                    </div>
                    {/* <div>
                        <ReactMic
                            record={isRecording}
                            className="sound-wave"
                            onStop={onStop}
                            onData={onData}
                            strokeColor="#000000"
                            backgroundColor="#FF4081" />
                        <button onClick={startRecording} disabled={isRecording}>Start</button>
                        <button onClick={stopRecording} disabled={!isRecording}>Stop</button>
                    </div> */}
                    <div className="ml-6 w-[600450px]">
                      <input
                      className="w-full rounded border p-2 focus:border-2 border-finalbluu focus:outline-none"
                      id="content"
                      placeholder="Message"
                      type="text"
                      value={userInput}
                      onChange={handleInputChange}>
                      </input>
                    </div>
                    <button
                    className="flex shrink-0 ml-5 items-center justify-center rounded-xl bg-finalbluu px-4 py-1 text-white hover:bg-finalred btn-animate transition-custom"
                    type="submit">
                      <span>SEND</span>
                      <span className="ml-2">
                        <svg
                        className="-mt-px h-4 w-4 rotate-45"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        xmlns="http://www.w3.org/2000/svg">
                          <path
                          d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2">
                          </path>
                        </svg>
                      </span>
                    </button>
                  </div>
                </form>
              </div>
            </section>
          ) : null}


          {/* 個人筆記本 */}
          {view === 'Notebook' ? (
            <section className="flex flex-col h-[95%] bg-gray-100 relative p-4">
              {/* 搜尋框 */}
              <div className="w-full max-w-lg mx-auto mb-4">
                <input
                type="text"
                placeholder="搜尋"
                className="w-[90%] px-4 py-2 ml-12 text-gray-700 bg-white border-2 rounded-md focus:outline-none focus:border-indigo-500"
                onChange={(e) => setSearchString(e.target.value)}>
                </input>
              </div>
              {/* 迭代 notebookContent 並為每條收藏的內容創建一個列表項目 */}
              <div className="bg-white rounded-md shadow-md overflow-y-auto overflow-x-hidden">
                {notebookContent
                .filter((content) => !searchString || (content.answer && content.answer.includes(searchString)))
                .map((content, index) => (
                  <div
                  key={index} 
                  className="p-4 border-b border-gray-300 relative bg-white rounded-md shadow-sm overflow-hidden">
                    {/* 編輯按鈕 */}
                    <button
                    className="text-finalbluu px-3 py-1 rounded hover:bg-finalbutter absolute top-2 right-16 transition duration-300 btn-animate transition-custom"
                    onClick={() => handleEditOrSave(index)}>
                      {editingIndex === index ? '完成' : '編輯'}
                    </button>
                    {/* 刪除按鈕 */}
                    <button
                    className="text-finalred px-3 py-1 rounded hover:bg-finalbutter absolute top-2 right-2 transition duration-300 btn-animate transition-custom"
                    onClick={() => handleDelete(index)}>
                      刪除
                    </button>
                    {editingIndex === index ? (
                      <div className="mt-8 bg-gray-100 p-4 border glowing-border rounded-md">
                        <label className="font-bold text-gray-600">
                          Q:
                        </label>
                        <input
                        type="text"
                        className="w-full p-2 mt-1 border rounded-md shadow-sm focus:outline-none"
                        value={editingItem.question}
                        onChange={(e) => updateEditingItem ('question', e.target.value)}>
                        </input>
                        <hr className="my-4"></hr>
                        <label className="font-bold text-gray-600">
                          A:
                        </label>
                        <textarea
                        className="w-full p-2 mt-1 border rounded-md shadow-sm focus:outline-none"
                        style={{ height: '200px' }}
                        value={editingItem.answer}
                        onChange={(e) => updateEditingItem('answer', e.target.value)}>
                        </textarea>
                      </div>

                    ) : (
                      <diiv>
                        <p
                        className="text-gray-700 mt-4 text-lg"
                        dangerouslySetInnerHTML={{
                          __html:
                          `<span className="font-bold">Q:</span>${highlightSearchString(content.question)}<br />
                          <span className="font-bold">A:</span>${highlightSearchString(content.answer)}`,
                        }}>
                        </p>
                        <p className="text-gray-500 mt-2 text-sm">
                          {highlightSearchString(content.date)}
                        </p>
                      </diiv>
                    )}
                  </div>
                ))}
              </div>

              {/* 返回按鈕 */}
              <div
              className="px-4 py-2 text-white bg-finalbluu rounded-md focus:outline-none cursor-pointer absolute top-4 left-4 btn-animate transition-custom"
              onClick={() => { setView('Chating'); }}>
                ←
              </div>
            </section>
          ) : null}


          {/* 資料管理 */}
          {view === 'Settings' ? (
            <section style={{backgroundColor: '#f0f0f0', padding: '20px', fontFamily: 'Arial, sans-serif'}}>
              
              <div style={{ textAlign: 'center' }}>
                
                {['Table1', 'Table2', 'Table3', 'Table4', 'Table5'].map((table) => (
                  <button
                  key={table}
                  className={`bg-finalbluu ${selectedTable === table ? 'selected' : ''}`}
                  style={{marginRight: '10px', marginBottom: '10px', padding: '5px 10px', color: '#fff', border: 'none', cursor: 'pointer'}}
                  onClick={() => handleTableSelect(table)}>
                    {table}
                  </button>
                  
                ))}
                <input
                  type="file"
                  accept=".csv, .docx, .txt" // 指定允许上传的文件类型
                  onChange={handleFileChange } // 处理文件上传的函数
                />
                <button onClick={handleUploadClick}>上傳</button>
              </div>
              <div style={{overflowY: 'auto', maxWidth: '100%', maxHeight: '750px'}}>
                {selectedTable && (
                  <section>
                    <table style={{width: '100%', borderCollapse: 'collapse', border: '1px solid #ccc'}}>
                      <thead>
                        <tr 
                        className="bg-finalbluu"
                        style={{ color: '#fff' }}>
                          <th>Name</th>
                          <th>Value</th>
                          <th>Actions</th> {/* 新增的列 */}

                        </tr>
                      </thead>
                      <tbody>
                        {tableData.map((item) => (
                          <tr 
                          key={item.id} 
                          style={{ borderBottom: '1px solid #ccc' }}>
                            <td style={{ padding: '8px', borderRight: '1px solid #ccc' }}>{item.name}</td>
                            <td style={{ padding: '8px', borderRight: '1px solid #ccc' }}>
                              {fullViewId === item.id ? item.value : item.value.substring(0, 10) + '...'}
                              <button
                              style={{
                                marginLeft: '10px',
                                padding: '2px 5px',
                                fontSize: '0.9em',
                                backgroundColor: '#f0f0f0',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                cursor: 'pointer',
                              }}
                              onClick={() => handleFullViewToggle(item.id)}>
                                {fullViewId === item.id ? '隱藏部分' : '查看完整'}
                              </button>
                            </td>
                            <td style={{ padding: '8px' }}>
                              <button
                              style={{
                                padding: '2px 5px',
                                backgroundColor: 'gray',
                                color: '#fff',
                                border: 'none',
                                borderRadius: '3px',
                                cursor: 'pointer',
                              }}
                              onClick={() => handleDeleteTable(item.id)}>
                                刪除
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </section>
                )}
              </div>
              <div
              className="px-4 py-2 text-white bg-finalbluu rounded-md focus:outline-none cursor-pointer absolute top-4 left-4 btn-animate transition-custom"
              onClick={() => {setView('Chating');}}>
                ←
              </div>
            </section>
          ) : null}
        </div>
      )}
      </div>
    </div>
    </>
  )
}

export default App;

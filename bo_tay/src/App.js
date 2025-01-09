import React, {useEffect, useRef, useState} from 'react';
import {Howl} from 'howler';
import { initNotifications, notify } from '@mycv/f8-notification';
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import * as faceDetection from '@tensorflow-models/face-detection';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';
import './App.css';
import soundURL from './assets/oi-ban-oi.mp3'

var sound = new Howl({
   src: [soundURL]
 });
 
const NOT_TOUCH_LABEL = 'not_touch';
const TOUCHED_LABEL = 'touched';
const TRAINING_TIMES = 50;
const TOP_CONFIDENCE =  0.8;

function App() {
   const video = useRef();
   const mobilenetModule = useRef();
   const classifier = useRef();
   const canPlaySound = useRef(true);
   const faceDetector = useRef(null);
   // Thêm state để kiểm tra có chạm hay không
   const [touched, setTouched] = useState(false);
   // Thêm state để kiểm tra khởi tạo hoàn tất
   const [isReady, setIsReady] = useState(false); 
   const [train1Complete, setTrain1Complete] = useState(false);
   const [train2Complete, setTrain2Complete] = useState(false);
   const [isTrained, setIsTrained] = useState(false);
   const [isRunning, setIsRunning] = useState(false);
   const [isTraining, setIsTraining] = useState(false);
   // Thêm state cho tiến trình huấn luyện
   const [trainingProgress, setTrainingProgress] = useState(0); 

   const init = async () => {
      try {
         await setupCamera();

         classifier.current = knnClassifier.create();
         mobilenetModule.current = await mobilenet.load(); 
         await initFaceDetection(); // Khởi tạo phát hiện khuôn mặt

         initNotifications({ cooldown: 3000 });
         setIsReady(true); // Đặt isReady thành true khi hoàn tất khởi tạo
      } catch (error) {
         console.error("Khởi tạo lỗi: ", error);
      }
   }

   // Khởi tạo mô hình phát hiện khuôn mặt
   const initFaceDetection = async () => {
      faceDetector.current = await faceDetection.createDetector(faceDetection.SupportedModels.MediaPipeFaceDetector, {
         runtime: 'tfjs',
         maxFaces: 1, // Chỉ kiểm tra một khuôn mặt
      });
   };

   // Hàm kiểm tra có khuôn mặt trong khung hình
   const detectFace = async () => {
      if (!faceDetector.current || !video.current) return false;

      const faces = await faceDetector.current.estimateFaces(video.current);
      return faces.length > 0;
   };

   const setupCamera = () => {
      return new Promise((resolve, reject) => {
         navigator.getUserMedia = navigator.getUserMedia || 
            navigator.webkitGetUserMedia ||
            navigator.mozGetUserMedia || 
            navigator.msGetUserMedia;

         if(navigator.getUserMedia) {
            navigator.getUserMedia(
               {video: true}, 
               stream => {
                  video.current.srcObject = stream;
                  video.current.addEventListener('loadeddata', resolve); // Thêm resolve ở đây để báo hiệu rằng camera đã được thiết lập thành công.
               }, error => {
                  console.error("Lỗi cài đặt camera:", error);
                  reject(error); // Thêm reject ở đây để báo lỗi khi có lỗi xảy ra.
               }
            );
         }else {
            reject(new Error("Không hỗ trợ getUserMedia trên trình duyệt này!"));
         }
      });
   }

   const train = async label => {
      setIsTraining(true); // Bắt đầu trạng thái huấn luyện
      console.log(`[${label}] đang train cho máy khuôn mặt của bạn`);

      for (let i = 0; i < TRAINING_TIMES; i++) {
         const faceDetected = await detectFace(); // Kiểm tra khuôn mặt
         if (!faceDetected) {
            console.warn('Không phát hiện khuôn mặt, vui lòng đảm bảo khuôn mặt ở trong khung hình.');
            alert('Không phát hiện khuôn mặt, vui lòng đảm bảo khuôn mặt ở trong khung hình.');
            setIsTraining(false);
            return;
         }

         // Cập nhật tiến trình
         const progress = parseInt(((i + 1) / TRAINING_TIMES) * 100);
         setTrainingProgress(progress);

         console.log(`Tiến trình ${parseInt(((i + 1) / TRAINING_TIMES) * 100)}%`);
         await training(label);
      }

      saveModel(); // Lưu mô hình sau khi huấn luyện
      console.log(`Train cho [${label}] đã hoàn  thành. Hình ảnh hiện tại ở:`, classifier.current.getNumClasses());

      // Đánh dấu hoàn thành Train1 hoặc Train2
      if (label === NOT_TOUCH_LABEL) {
         setTrain1Complete(true);
      } else if (label === TOUCHED_LABEL) {
         setTrain2Complete(true);
      }

      checkIfTrained(); 
      setIsTraining(false); // Kết thúc trạng thái huấn luyện
   };
   
   /**
    * Bước 1: Train cho máy khuôn mặt không chạm tay
    * Bước 2: Train cho máy khuôn mặt có chạm tay
    * Bước 3: Lấy hình ảnh hiện tại, phân tích và so sánh với data đã học
    * ==> Nếu matching với data khuôn mặt chạm ==> cảnh báo
    * @param {*} label 
    * @returns 
    */

   const saveModel = () => {
      if (classifier.current) {
         const dataset = classifier.current.getClassifierDataset();
         const datasetJSON = JSON.stringify(Object.fromEntries(Object.entries(dataset).map(([label, data]) => [label, Array.from(data.dataSync())])));
         localStorage.setItem('knnClassifierDataset', datasetJSON);
         console.log('Đã lưu vào localStorage!');
      }
   };

   const loadModel = () => {
      const datasetJSON = localStorage.getItem('knnClassifierDataset');
      if (datasetJSON) {
         try {
            const dataset = JSON.parse(datasetJSON);
            const tensorDataset = Object.fromEntries(
               Object.entries(dataset).map(([label, array]) => [
                  label,
                  tf.tensor(array, [array.length / 1024, 1024]), // Chuyển đổi về tensor
               ])
            );
            classifier.current.setClassifierDataset(tensorDataset);
         } catch (error) {
            console.error('Không tải được dữ liệu:', error);
            clearModel(); // Xóa dữ liệu lỗi
            alert('Không có dữ liệu. Thử lại!');
         }
         
      } else {
         console.log('Không tìm thấy dữ liệu đã lưu!');
      }
   };   
   
   const clearModel = () => {
      localStorage.removeItem('knnClassifierDataset');
      console.log('Đã xóa dữ liệu khỏi localStorage!');

      if (classifier.current) {
         classifier.current.clearAllClasses();
         console.log('Xóa tập dữ liệu phân loại!');
      }
      
      // Đặt lại trạng thái huấn luyện
      setIsTrained(false); 
      setTrain1Complete(false);
      setTrain2Complete(false);
      setTouched(false);
   };
   
   const checkIfTrained = () => {
      if (classifier.current.getNumClasses() >= 2) {
         setIsTrained(true);
      } else {
         setIsTrained(false);
      }
   }   
   
   const training = label => {
      return new Promise(async resolve => {
         const embedding = mobilenetModule.current.infer(
            video.current,
            true
         );
         classifier.current.addExample(embedding, label);
         await sleep(100);
         resolve();
      });
   };

   const run = async () => {   
      // Kiểm tra xem classifier đã có ví dụ hay chưa
      if (!classifier.current || classifier.current.getNumClasses() < 2) {
         alert('Không có đủ dữ liệu. Hãy train lại dữ liệu');
         return;
      }

      setIsRunning(true); // Đánh dấu là hệ thống đang chạy
      // Quá trình chạy liên tục (vòng lặp nhận diện)
      const embedding = mobilenetModule.current.infer(video.current, true);
      const result = await classifier.current.predictClass(embedding);
   
      if ((result.label === TOUCHED_LABEL || result.classIndex !== 0) && result.confidences[result.label] > TOP_CONFIDENCE) {
         setTouched(true);
         if (canPlaySound.current) {
            canPlaySound.current = false;
            sound.play();
         }
         notify('BỎ TAY RA!', { body: 'Bạn vừa chạm tay vào mặt!' });
      } else {
         setTouched(false);
      }
   
      await sleep(200);
      setIsRunning(false); // Khi quá trình hoàn tất, đặt lại trạng thái là không chạy
      run();
   };
   

   const sleep = (ms = 0) => {
      return new Promise(resolve => setTimeout(resolve, ms))
   }

   useEffect(() => {
      init();
      sound.on('end', function(){
         canPlaySound.current = true;
       });
      return () => {
      }
   // eslint-disable-next-line react-hooks/exhaustive-deps
   }, []);

   return (
      <div className={`main ${touched ? 'touched' : ''}`}>
         <h1>DEMO IMAGE RECOGNITION</h1>
         <video 
            ref={video}
            className="video"
            autoPlay
         />

         <div className="control">         
            {!train1Complete && (
               <button className="btn" onClick={() => train(NOT_TOUCH_LABEL)} disabled={!isReady || isTraining}>Bắt Đầu</button>
            )}                            
            {train1Complete && !train2Complete && (
               <button className="btn" onClick={() => train(TOUCHED_LABEL)} disabled={isTraining}>Bắt Đầu</button>
            )} 
            {train2Complete && !isRunning && (
               <button className="btn" onClick={() => run()} disabled={false}>Chạy</button>
            )}
            {train2Complete && (
               <button className="btn" onClick={() => clearModel()} disabled={false}>Xóa Dữ Liệu</button>
            )}

            <h2 className="status">
               {!isReady && <p>Đang khởi động.....vui lòng chờ</p>}
               {isReady && !train1Complete && !isTraining && <p>Quay video không chạm tay lên mặt.</p>}
               {isReady && !train1Complete && isTraining && <p>Đang chạy...{trainingProgress}%</p>}
               {train1Complete && !train2Complete && !isTraining && <p>Quay video đưa tay gần lên mặt.</p>}
               {train1Complete && !train2Complete && isTraining && <p>Đang chạy...{trainingProgress}%</p>}
               {train2Complete && isTrained && !isRunning && <p>Chạy hệ thống hoặc xóa dữ liệu nếu muốn</p>}
               {isRunning && <p>Hệ thống đang chạy...</p>}
            </h2>

         </div>

      </div>
   );
   }

export default App;

# --- 保留所有 KKT 模組相關的導入 ---
from KKT_Module.KKT_Module import kgl
from KKT_Module.KKT_Module.DataReceive.Core import Results
from KKT_Module.KKT_Module.DataReceive.DataReceiver import MultiResult4168BReceiver
from KKT_Module.KKT_Module.FiniteReceiverMachine import FRM
from KKT_Module.KKT_Module.SettingProcess.SettingConfig import SettingConfigs
from KKT_Module.KKT_Module.SettingProcess.SettingProccess import SettingProc
from KKT_Module.KKT_Module.GuiUpdater.GuiUpdater import Updater

import sys
import os
import time
import serial   # 引入序列通訊庫
import numpy as np 
from collections import deque 

import tensorflow as tf
from tensorflow.keras import layers, models

# =========================================================
# === 全域設定與 Helper 函式區 ===
# =========================================================

# KKT Library 路徑設定
current_dir = os.path.dirname(os.path.abspath(__file__))
library_path = os.path.join(current_dir, 'Library')
if library_path not in sys.path:
    sys.path.insert(0, library_path)

# 手勢對應表 (常數，維持全域即可)
INVERSE_LABEL_MAP = {
    0: 'background',
    1: 'open',
    2: 'close',
    3: 'turn',
    999: 'keep',
}

def init_serial_connection():
    """初始化 Arduino 連線並回傳 serial 物件"""
    port_number = "COM8"  # 請確認您的 COM Port
    ser = None
    try:
        ser = serial.Serial(port_number, 9600, timeout=1)
        time.sleep(2) # 等待 Arduino 重啟連線穩定
        print("INFO: Arduino connected successfully via Serial.")
    except Exception as e:
        print(f"WARNING: Could not connect to Arduino. Details: {e}")
        ser = None
    return ser

def connect():
    """連接 KKT 設備"""
    try:
        device = kgl.ksoclib.connectDevice()
        print(f"INFO: Device connected successfully. ID: {device}")
    except Exception as e:
        print(f"ERROR: Connection Failed. Please check device and try again. Details: {e}")
        sys.exit(1)

def setProperty(obj: object, **kwargs):
    """設定 KKT 屬性"""
    print(f'==== Set properties in {obj.__class__.__name__} ====')
    for k, v in kwargs.items():
        if not hasattr(obj, k):
            print(f'Attribute "{k}" not in {obj.__class__.__name__}.')
            continue
        obj.__setattr__(k, v)
        print(f'Attribute "{k}", set "{v}"')

def setScript(setting_name: str):
    """設定 KKT 腳本"""
    ksp = SettingProc()
    setting_config = SettingConfigs()
    setting_config.Chip_ID = kgl.ksoclib.getChipID().split(' ')[0]
    setting_config.Processes = [
        'Reset Device', 'Gen Process Script', 'Gen Param Dict', 'Get Gesture Dict',
        'Set Script', 'Run SIC', 'Phase Calibration', 'Modulation On'
    ]
    setting_config.setScriptDir(f'{setting_name}')
    ksp.startUp(setting_config)

# =========================================================
# === 模型定義區 ===
# =========================================================

def model_struct(LSTM_units, dense_hidden_units, dropout_rate, num_classes, time_steps, width, height):
    input_shape = (time_steps, width, height, 1)
    sequential_model = tf.keras.Sequential([
        layers.TimeDistributed(layers.Conv2D(32, (7, 7), padding='same', activation='relu'),input_shape=input_shape),
        layers.TimeDistributed(layers.MaxPooling2D((3, 3))),
        layers.TimeDistributed(layers.Conv2D(32, (5, 5), padding='same', activation='relu')),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', activation='relu')),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.Flatten()),
        layers.LSTM(LSTM_units),
        layers.Dense(dense_hidden_units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    return sequential_model

def restruct_model(weight_path):
    # 模型參數設定
    time_steps = 20
    height = 32
    width = 32
    num_classes= 4
    LSTM_units = 64
    dense_hidden_units = 64
    dropout_rate = 0.2
    
    struct_m = model_struct(LSTM_units, dense_hidden_units, dropout_rate, num_classes, time_steps, width, height)
    try:
        struct_m.load_weights(weight_path)  
    except:
        print("\n[error] : check inference file parameters or path.\n")
        exit()
    return struct_m

class ExampleGestureModel:
    """模型包裝類別"""
    def __init__(self, input_shape=(20, 32, 32)):
        try:
            # 請修改為您的權重檔路徑
            weight = "./model/gesture_fans_64_0.2_64_ep_80.weights.h5"
        except:
            print("[error] : load weight fail")
            weight = "" # 避免 crash，實際應處理錯誤
        
        self.model = restruct_model(weight)
        self.input_shape = input_shape
        print(f"INFO: 模型已載入，預期輸入形狀: {input_shape}")

    def predict(self, window_data_batch):
        predictions = self.model.predict(window_data_batch)
        return predictions

# =========================================================
# === 新增的核心類別：手勢動作控制器 (GestureActionController) ===
# =========================================================

class GestureActionController:
    """
    負責管理手勢狀態 (Open/Turn/Counter) 並控制 Arduino。
    原本的全域變數與 process_window_data 邏輯都封裝在此。
    """
    def __init__(self, arduino_serial_obj):
        # 1. 接收外部傳入的 serial 物件
        self.arduino = arduino_serial_obj
        
        # 2. 初始化狀態變數 (取代原本的全域變數)
        self.open_state = 0
        self.turn_state = 0
        self.turn_counter = 0
        
        print("INFO: GestureActionController initialized.")
        print(f"Initial State -> Open: {self.open_state}, Turn: {self.turn_state}")

    def process_gesture(self, window_data, model_instance):
        """
        核心邏輯方法。
        對應原本的 process_window_data 函式。
        """
        # 1. 準備模型輸入
        window_batch = np.expand_dims(window_data, axis=0)
        window_batch = np.expand_dims(window_batch, axis=-1) # 若模型需要 channel 維度

        # 2. 執行模型推論
        predictions = model_instance.predict(window_batch)
        
        # 3. 解讀輸出
        pred_probabilities = predictions[0]
        predicted_index = np.argmax(pred_probabilities)
        confidence = pred_probabilities[predicted_index]

        # 門檻值過濾
        if confidence < 0.85:
            predicted_index = 0  # background

        # -------------------------------------------------------------
        # A. 重置計數器邏輯 (使用 self.turn_counter)
        # -------------------------------------------------------------
        if predicted_index != 3:
            self.turn_counter = 0

        # -------------------------------------------------------------
        # B. 處理 Turn (手勢 3)
        # -------------------------------------------------------------
        if predicted_index == 3:
            self.turn_counter += 1
            
            if self.turn_counter == 3:
                print("turn state change")
                if self.turn_state == 0:
                    self.turn_state = 1
                    print("turn_state 0 => 1 (Start Turning)")
                else:
                    self.turn_state = 0
                    print("turn_state 1 => 0 (Stop Turning)")
            
            elif self.turn_counter < 3:
                predicted_index = 999
                print(f"Turn gesture accumulating... {self.turn_counter}/3")
            
            else:
                # turn_counter > 3 (避免重複觸發)
                predicted_index = 999

        # -------------------------------------------------------------
        # C. 處理 Open (手勢 1)
        # -------------------------------------------------------------
        if predicted_index == 1:
            if self.open_state == 1:
                predicted_index = 999
                print(">>> Predicted Gesture: keep open (Already Open)")
            else:
                self.open_state = 1
                print("open_state 0 => 1 (Fan ON)")

        # -------------------------------------------------------------
        # D. 處理 Close (手勢 2)
        # -------------------------------------------------------------
        if predicted_index == 2:
            if self.open_state == 0 and self.turn_state == 0:
                predicted_index = 999
            else:
                self.open_state = 0
                if self.turn_state == 1:
                    self.turn_state = 0
                    print("turn_state 1 => 0 (Forced Stop by Close)")
                print("open_state => 0 (Fan OFF)")

        # 將索引轉換回手勢名稱
        predicted_gesture = INVERSE_LABEL_MAP[predicted_index]

        # 4. 印出結果
        print(f"*** Gesture Prediction ***")
        print(f">>> Predicted: {predicted_gesture} (Idx: {predicted_index}) | Confident: {confidence:.2f}")
        print(f"State \nOpen: {self.open_state} \nTurn: {self.turn_state}")
        print("\n")

        # 5. 發送 Arduino 指令 (使用 self.arduino)
        if self.arduino is not None and self.arduino.is_open:
            command_char = None
            
            if predicted_gesture == 'open':
                print("Command: Open Fans")
                command_char = 'O'
            elif predicted_gesture == 'close':
                print("Command: Close Fans")
                command_char = 'C'
            elif predicted_gesture == 'turn' and self.turn_counter == 3:
                if self.turn_state == 1: # 剛切換成 1 (Start)
                    print("Command: Turn Fans")
                    command_char = 'T'
                elif self.turn_state == 0: # 剛切換成 0 (Stop)
                    print("Command: Stop Turning")
                    command_char = 'S'
            
            if command_char:
                try:
                    self.arduino.write(command_char.encode())
                    print(f"    -> Sent to Arduino: {command_char}")
                except Exception as e:
                    print(f"    -> Serial Write Error: {e}")

# =========================================================
# === 資料處理器 (維持原狀，只需傳入 callback) ===
# =========================================================

class CustomDataProcessor(Updater):
    """
    資料更新器：負責緩衝 (Buffering) 和 滑動窗口 (Sliding Window)。
    """
    def __init__(self, data_type, callback_function, model_instance, window_size, stride_step):
        super().__init__()
        self.show_data_type = data_type
        self.window_size = window_size
        self.stride_step = stride_step
        self.buffer = deque(maxlen=self.window_size)
        self.frames_since_last_predict = self.stride_step
        self.process_func = callback_function # 這裡會儲存 Controller 的 method
        self.model = model_instance

    def update(self, res: Results):
        result_data = None
        if self.show_data_type == 'feature_map':
            result_data = res['feature_map'].data
        elif self.show_data_type == 'raw_data':
            result_data = res['raw_data'].data
        else:
            return 

        result_data = result_data[0]
        self.buffer.append(result_data)
        self.frames_since_last_predict += 1

        if len(self.buffer) < self.window_size:
            return

        if self.frames_since_last_predict >= self.stride_step:
            self.frames_since_last_predict = 0
            window_data = np.array(self.buffer)
            try:
                # 呼叫 Controller 的方法
                self.process_func(window_data, self.model)
            except Exception as e:
                print(f"ERROR inside callback: {e}")

# =========================================================
# === 主程式執行區 ===
# =========================================================

if __name__ == '__main__':

    WINDOW_SIZE = 20
    STRIDE_STEP = 5

    # 1. 初始化 Serial (取得物件)
    my_arduino_serial = init_serial_connection()

    # 2. 初始化動作控制器 (傳入 Serial 物件)
    #    所有的狀態 (open_state 等) 現在都活在這個物件裡
    gesture_controller = GestureActionController(my_arduino_serial)

    # 3. 載入模型
    my_model = ExampleGestureModel(input_shape=(WINDOW_SIZE, 32, 32))

    # 4. KKT 腳本設定
    setting_file = r'K60168-Test-00256-008-v0.0.8-20230717_60cm'
    kgl.setLib()
    connect()
    setScript(setting_file)

    data_type = 'feature_map'
    if data_type == 'raw_data':
        kgl.ksoclib.writeReg(0, 0x50000504, 5, 5, 0)
    elif data_type == 'feature_map':
        kgl.ksoclib.writeReg(1, 0x50000504, 5, 5, 0)

    # 5. 設定 Processor
    #    注意 callback_function 傳入的是 controller.process_gesture
    processor = CustomDataProcessor(
        data_type=data_type,
        callback_function=gesture_controller.process_gesture, # <--- 關鍵修改
        model_instance=my_model,
        window_size=WINDOW_SIZE,
        stride_step=STRIDE_STEP
    )

    receiver = MultiResult4168BReceiver()
    receiver_args = { "actions": 1, "rbank_ch_enable": 7, "read_interrupt": 0, "clear_interrupt": 0 }
    setProperty(receiver, **receiver_args)

    FRM.setReceiver(receiver)
    FRM.setUpdater(processor)
    FRM.trigger()

    try:
        FRM.start()
        print("\n*** Real-time Gesture Recognition Started (Class-based) ***")
        print("Press Ctrl+C to stop.")
        
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nKeyboard Interrupt. Stopping...")
        
        # 透過控制器關閉 Arduino
        if my_arduino_serial is not None and my_arduino_serial.is_open:
            try:
                print(" -> Sending STOP command ('C')...")
                my_arduino_serial.write('C'.encode())
                time.sleep(0.5)
                my_arduino_serial.close()
                print(" -> Arduino Serial disconnected.")
            except Exception as e:
                print(f" -> Error sending stop command: {e}")

    finally:
        FRM.stop()
        kgl.ksoclib.closeDevice()
        print("Device closed.")
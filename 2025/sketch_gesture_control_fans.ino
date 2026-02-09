#include <Servo.h>

// --- 硬體接腳設定 (不可修改) ---
const int PIN_FAN = 13;    // 風扇馬達 (繼電器)
const int PIN_SERVO = 9;   // 伺服馬達

Servo myservo;

// --- 狀態變數 ---
bool isFanRunning = false;      // 風扇狀態: true=轉, false=停
bool isServoRotating = false;   // 伺服馬達狀態: true=來回轉, false=停

// --- 伺服馬達控制變數 ---
int servoPos = 90;              // 目前角度
int servoStep = 1;              // 移動步伐 (1代表正轉, -1代表反轉)
unsigned long lastServoTime = 0;
const int SERVO_INTERVAL = 15;  // 轉動速度 (毫秒): 數值越小轉越快

void setup() {
  // 初始化接腳
  pinMode(PIN_FAN, OUTPUT);
  myservo.attach(PIN_SERVO);
  
  // 初始化 Serial，必須與 Python 設定的 9600 一致
  Serial.begin(9600);

  // 初始狀態：全部停止
  digitalWrite(PIN_FAN, LOW);
  //myservo.write(90); // 伺服馬達回歸中間值
}

void loop() {
  // ==========================================
  // 1. 讀取 Python 傳來的指令
  // ==========================================
  if (Serial.available() > 0) {
    char command_char = Serial.read();

    // 根據指令切換狀態
    if (command_char == 'O') {
      // 'O': 風扇馬達持續運作
      isFanRunning = true;
    } 
    else if (command_char == 'C') {
      // 'C': 風扇馬達及伺服馬達停止
      isFanRunning = false;
      isServoRotating = false;
    } 
    else if (command_char == 'T') {
      // 'T': 伺服馬達持續運轉
      isServoRotating = true;
    } 
    else if (command_char == 'S') {
      // 'S': 伺服馬達停止運作
      isServoRotating = false;
    }
  }

  // ==========================================
  // 2. 執行風扇控制
  // ==========================================
  if (isFanRunning) {
    digitalWrite(PIN_FAN, LOW); // 繼電器通電 (風扇轉)
  } else {
    digitalWrite(PIN_FAN, HIGH);  // 繼電器斷電 (風扇停)
  }

  // ==========================================
  // 3. 執行伺服馬達控制 (非阻塞式掃描)
  // ==========================================
  if (isServoRotating) {
    unsigned long currentTime = millis();
    
    // 每隔 SERVO_INTERVAL 毫秒移動一次，不使用 delay() 避免卡住
    if (currentTime - lastServoTime >= SERVO_INTERVAL) {
      lastServoTime = currentTime;

      // 更新角度
      servoPos += servoStep;

      // 檢查邊界：轉到底就往反方向轉
      if (servoPos >= 180) {
        servoPos = 180;
        servoStep = -1; // 改為減少角度 (反轉)
      } 
      if (servoPos <= 0) {
        servoPos = 0;
        servoStep = 1;  // 改為增加角度 (正轉)
      }

      myservo.write(servoPos);
    }
  }
}
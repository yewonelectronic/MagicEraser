//persondetection 사용 위한 라이브러리
#include <TensorFlowLite.h>
#include "main_functions.h"
#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
//인터룹트 사용 위한 라이브러리
#include "NRF52_MBED_TimerInterrupt.h"
#include "NRF52_MBED_ISR_Timer.h"

///mp3 플레이어를 위한 라이브러리
#include <DFRobotDFPlayerMini.h>
DFRobotDFPlayerMini MP3Player;
UART mySoftwareSerial(digitalPinToPinName(3), digitalPinToPinName(2),NC, NC); // RX, TX

//초음파 센서와 서보모터 사용위한 라이브러리
#include <NewPing.h>
#include <Servo.h>
//초음파 센서 핀번호 및 max 거리 설정
#define TRIG_PIN 4
#define ECHO_PIN 5
#define MAX_DISTANCE 200
NewPing sonar(TRIG_PIN, ECHO_PIN, MAX_DISTANCE);
//서보모터 선언
Servo servo;
int pos = 40;

// LED 핀번호 설정
#define LED_RED D9
#define LED_GREEN D10
// stop, start 버튼 핀 배정
#define StopButtonPin A1 // blue
#define StartButtonPin A2 // yellow
// ISR에서 변화한 state값을 보존하기 위해 volatile로 작성.
volatile byte state = LOW;
volatile byte state_prv = LOW;
volatile byte state_btn = HIGH;
volatile byte state_cam = LOW;
int count=0;

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

constexpr int kTensorArenaSize = 136 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
}  
void setup() {
  servo.attach(6); // 서보모터 핀 6번
  Serial.begin(9600);

  mySoftwareSerial.begin(9600);//MP3를 위한 sortwareSerial
  Serial.println(F("connectec"));
  if (!MP3Player.begin(mySoftwareSerial)) { // MP3 모듈을 초기화합니다. 초기화에 실패하면 오류를 발생시킵니다.
    Serial.println(F("Unable to begin:"));
    Serial.println(F("1.Please recheck the connection!"));
    Serial.println(F("2.Please insert the SD card!"));
    while (true);
  }
  delay(1);
  MP3Player.volume(30);  // 볼륨을 조절합니다. 0~30까지 설정이 가능합니다.>> 현재는 20으로 volume 설정

  //버튼을 눌러 인터럽트 핀인 SWTICH_PIN의 값이 low에서 high로 변화할 때 ISR이 실행되도록 한다.
  attachInterrupt(digitalPinToInterrupt(StopButtonPin), Stop, RISING);
  attachInterrupt(digitalPinToInterrupt(StartButtonPin), Start, RISING);

  //3개의 LED를 OUTPUT으로 설정한다.
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);

  // 버튼 핀 설정
  pinMode(StopButtonPin, INPUT);
  pinMode(StartButtonPin, INPUT);

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interpreter(
    model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;


  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }
  input = interpreter->input(0);
}

//각종 state 정리
//state: 동작을 해야하면 high
//state_prv:이전에 동작했으면 high/ 동작하지 않았으면 low
//state_btn: stop: low/ start: high
//state_cam: 사람이 인식되면 high/ 인식되지 않으면 low

//////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
void loop() {
  ////갑작스럽게 시작될 경우를 대비하여 처음 loop가 돌아갈 때에만 delay(1000)
  if (count==0) delay(3000);
  count =1;
  // Get image from provider.
  if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
                            input->data.int8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
  }

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  int8_t person_score = output->data.uint8[kPersonIndex];
  int8_t no_person_score = output->data.uint8[kNotAPersonIndex];
  RespondToDetection(error_reporter, person_score, no_person_score);

  if (person_score > 80) { //person_score > 80
    state_cam = HIGH; // 사람이 인식됨
  } else state_cam = LOW; // 사람이 인식되지 않음.

  if (state_btn == LOW && state_prv == HIGH) { // 이전에 동작하고 있었는데 stop button 눌릴 때>> 동작 중지
    MP3Player.play(4); // mp3 004 //작동을 중지합니다.
    delay(7000);
    state = LOW; // state LOW로 변경
    state_prv = LOW;
    digitalWrite(LED_RED, HIGH); //빨간색 LED를 ON
    digitalWrite(LED_GREEN, LOW);//초록색 LED를 OFF
  }

  if (state_btn == HIGH && state_prv == LOW && state_cam==LOW) { // 이전에 동작하고 있지 않은데 START 버튼이 눌릴 때>> 동작 시작
    state = HIGH; // HIGH
  }

  if (state_prv == LOW && state_cam == LOW && state_btn == HIGH) { // 이전에 동작하지 않을 때 사람이 인식되지 않으면>> 사람이 인식되지 않습니다. MP3 1출력
    // mp3 001
    MP3Player.play(1);
    state = HIGH;
    Serial.println("mp3 001"); // 사람이 인식되지 않습니다.
    delay(13000);
    if (state == HIGH){ //stop버튼이 눌리지 않으면
      Serial.println("mp3 003"); 
      // mp3 003
      MP3Player.play(3); // 작동을 시작합니다.
      delay(6000);
    }
  }
  
  if (state_cam == HIGH){//사람이 인식되면
    state = LOW;//동작하지 않음.
  }

  if (state_btn == HIGH && state_prv == LOW && state_cam==LOW) { // 이전에 동작하고 있지 않은데 START 버튼이 눌릴 때>> 동작 시작(interrupt를 통해 중간에 state 바뀌는 것에 대비)
    state = HIGH; // 동작함.
  }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//최종적인 동작 if문//
  if (state == LOW) {//state LOW이면 빨간색 LED ON
    digitalWrite(LED_RED, HIGH); 
    digitalWrite(LED_GREEN, LOW);
  }
  if (state == HIGH) {//state HIGH이면 초록색 LED ON
    digitalWrite(LED_GREEN, HIGH);
    digitalWrite(LED_RED, LOW);
    
    //칠판 지우는 서보모터 동작
    for (pos = 40; pos <= 140; pos += 1) {
      servo.write(pos);
      delay(7);
    }
    for (pos = 140; pos >= 40; pos -= 1) {
      servo.write(pos);
      delay(7);
    }
    //초음파 센서를 통해 거리를 재고 10 이하이면 위험 알림
    uint8_t so=sonar.ping_cm();
    if ((so < 10) &&(so > 0)) {
      // mp3 002
      MP3Player.play(2); 
      Serial.println("mp3 002");//위험합니다. 매직 지우개가 작동중입니다.
      delay(5000);
    }
  }
  state_prv = state;//현재 상태를 이전 상태변수(state_prv)에 저장한다.
}

//버튼 스위치 input을 interrupt로 활용하여 state를 바꾼다.
void Stop()
{
  // 중지
  state_btn = LOW;

}

void Start()
{
  //시작
  state_btn = HIGH;
}
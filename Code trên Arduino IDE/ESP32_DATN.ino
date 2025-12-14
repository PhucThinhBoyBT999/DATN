#define BLYNK_TEMPLATE_ID "TMPL6jrnSEcw5"
#define BLYNK_TEMPLATE_NAME "Quan Ly Phong Thi Nghiem"
#define BLYNK_AUTH_TOKEN "dRetcrvdh9fU4oY6Fd88XwqpBCCXNJ_5"
#define BLYNK_PRINT Serial

#include "config.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <BlynkSimpleEsp32.h>
#include <ArduinoJson.h>
#include <NTPClient.h>
#include <WiFiUdp.h>
#include <EEPROM.h>

char ssid[] = "DTI Studio";
char pass[] = "khongbiet";

const char *GOOGLE_USERS_URL = "https://script.google.com/macros/s/AKfycbxX3sWzaTqYUfEfOXxgaTFvpt4El9pOfIRl8uy006DgPbpS3osfx6V14zHcMRJ03ull/exec?action=getUsers";
const char *GOOGLE_LOG_URL = "https://script.google.com/macros/s/AKfycbxX3sWzaTqYUfEfOXxgaTFvpt4El9pOfIRl8uy006DgPbpS3osfx6V14zHcMRJ03ull/exec";

#define VPIN_LED_CONTROL      V0
#define VPIN_FAN_CONTROL      V1
#define VPIN_ROOM_COUNT       V3
#define VPIN_SCHEDULE_MODE    V6
#define VPIN_LAST_PERSON      V7
#define VPIN_TOTAL_IN         V8
#define VPIN_TOTAL_OUT        V9
#define VPIN_FACE_ID          V14
#define VPIN_RELOAD_STUDENTS  V20
#define VPIN_SCHEDULE_ON      V12
#define VPIN_SCHEDULE_OFF     V13

#define MAX_STUDENTS        50
#define MAX_PEOPLE_IN_ROOM  20
#define AUTH_TIMEOUT        20000
#define EEPROM_SIZE         512
#define MAGIC_NUMBER        0xABCD1234

struct Student {
  int id;
  String name;
  String mssv;
  String rfid;
  int faceID;
};

enum AuthState { AUTH_IDLE, AUTH_FACE_DETECTED, AUTH_COMPLETED };
enum AuthSource { SOURCE_NONE, SOURCE_RFID, SOURCE_FACE };

Student students[MAX_STUDENTS];
int studentCount = 0;
String peopleInRoom[MAX_PEOPLE_IN_ROOM];
int peopleCount = 0;
int totalIn = 0, totalOut = 0;

AuthState authState = AUTH_IDLE;
AuthSource authSource = SOURCE_NONE;
int detectedFaceID = 0;
String detectedName = "";
unsigned long authStartTime = 0;

String currentTime, currentDate;
int hh, mm, ss, da, mo, yr;
unsigned long lastTimeUpdate = 0, lastWiFiCheck = 0, lastBlynkCheck = 0;
unsigned long doorOpenTime = 0, lcdDisplayTime = 0;
bool doorIsOpen = false, lcdShowingInfo = false;

bool scheduleEnabled = false;
int scheduleON_hour = 7, scheduleON_min = 0;
int scheduleOFF_hour = 17, scheduleOFF_min = 30;
bool scheduleTurnedOn = false, scheduleTurnedOff = false;

WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP, "asia.pool.ntp.org", 3600 * 7);
BlynkTimer timer;

String rfidToString(byte *buffer, byte bufferSize) {
  String rfid = "";
  for (byte i = 0; i < bufferSize; i++) {
    rfid += String(buffer[i] < 0x10 ? "0" : "");
    rfid += String(buffer[i], HEX);
  }
  rfid.toUpperCase();
  return rfid;
}

String urlEncode(String str) {
  String encoded = "";
  for (unsigned int i = 0; i < str.length(); i++) {
    char c = str.charAt(i);
    if (c == ' ') encoded += "%20";
    else if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || 
             (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~') 
      encoded += c;
    else {
      encoded += "%";
      if (c < 16) encoded += "0";
      encoded += String(c, HEX);
    }
  }
  return encoded;
}

void updateTime() {
  timeClient.update();
  time_t epochTime = timeClient.getEpochTime();
  struct tm *ptm = gmtime((time_t *)&epochTime);
  hh = timeClient.getHours();
  mm = timeClient.getMinutes();
  ss = timeClient.getSeconds();
  da = ptm->tm_mday;
  mo = ptm->tm_mon + 1;
  yr = ptm->tm_year + 1900;
  char temp[20];
  sprintf(temp, "%02d:%02d:%02d", hh, mm, ss);
  currentTime = String(temp);
  sprintf(temp, "%02d/%02d/%04d", da, mo, yr);
  currentDate = String(temp);
}

void saveToEEPROM() {
  EEPROM.writeInt(0, peopleCount);
  EEPROM.writeInt(4, totalIn);
  EEPROM.writeInt(8, totalOut);
  EEPROM.writeInt(12, da);
  EEPROM.writeInt(16, MAGIC_NUMBER);
  int addr = 20;
  EEPROM.writeInt(addr, peopleCount);
  addr += 4;
  for (int i = 0; i < peopleCount && i < MAX_PEOPLE_IN_ROOM; i++) {
    EEPROM.writeInt(addr, peopleInRoom[i].toInt());
    addr += 4;
  }
  EEPROM.commit();
}

void loadFromEEPROM() {
  int magic = EEPROM.readInt(16);
  if (magic == MAGIC_NUMBER) {
    peopleCount = EEPROM.readInt(0);
    totalIn = EEPROM.readInt(4);
    totalOut = EEPROM.readInt(8);
    int lastDay = EEPROM.readInt(12);
    if (peopleCount < 0 || peopleCount > MAX_PEOPLE_IN_ROOM) peopleCount = 0;
    if (totalIn < 0) totalIn = 0;
    if (totalOut < 0) totalOut = 0;
    if (lastDay == da) {
      int addr = 20;
      int count = EEPROM.readInt(addr);
      addr += 4;
      if (count > 0 && count <= MAX_PEOPLE_IN_ROOM) {
        for (int i = 0; i < count; i++) {
          peopleInRoom[i] = String(EEPROM.readInt(addr));
          addr += 4;
        }
      }
    } else {
      totalIn = totalOut = peopleCount = 0;
      saveToEEPROM();
    }
  } else {
    peopleCount = totalIn = totalOut = 0;
    saveToEEPROM();
  }
}

void loadStudentsFromGoogleSheets() {
  if (WiFi.status() != WL_CONNECTED) return;
  HTTPClient http;
  WiFiClientSecure client;
  client.setInsecure();
  client.setTimeout(15000);
  http.begin(client, GOOGLE_USERS_URL);
  http.setTimeout(15000);
  http.setFollowRedirects(HTTPC_FORCE_FOLLOW_REDIRECTS);
  int httpCode = http.GET();
  
  if (httpCode == 200 || httpCode == 302) {
    String payload = http.getString();
    DynamicJsonDocument doc(4096);
    DeserializationError error = deserializeJson(doc, payload);
    if (!error && doc["success"].as<bool>()) {
      JsonArray users = doc["data"]["users"].as<JsonArray>();
      studentCount = 0;
      for (JsonObject user : users) {
        if (studentCount >= MAX_STUDENTS) break;
        students[studentCount].id = user["id"];
        students[studentCount].name = user["name"].as<String>();
        students[studentCount].mssv = user["mssv"].as<String>();
        students[studentCount].rfid = user["rfid"].as<String>();
        students[studentCount].faceID = user["faceID"];
        studentCount++;
      }
    }
  }
  http.end();
  client.stop();
}

Student* findStudentByFaceID(int faceID) {
  for (int i = 0; i < studentCount; i++) {
    if (students[i].faceID == faceID) return &students[i];
  }
  return nullptr;
}

Student* findStudentByRFID(String rfid) {
  for (int i = 0; i < studentCount; i++) {
    if (students[i].rfid.equalsIgnoreCase(rfid)) return &students[i];
  }
  return nullptr;
}

bool isPersonInRoom(int faceID) {
  for (int i = 0; i < peopleCount; i++) {
    if (peopleInRoom[i].toInt() == faceID) return true;
  }
  return false;
}

void addPersonToRoom(int faceID) {
  if (peopleCount >= MAX_PEOPLE_IN_ROOM) return;
  if (!isPersonInRoom(faceID)) peopleInRoom[peopleCount++] = String(faceID);
}

void removePersonFromRoom(int faceID) {
  for (int i = 0; i < peopleCount; i++) {
    if (peopleInRoom[i].toInt() == faceID) {
      for (int j = i; j < peopleCount - 1; j++) peopleInRoom[j] = peopleInRoom[j + 1];
      peopleCount--;
      return;
    }
  }
}

void logToGoogleSheets(Student* student, String action) {
  if (WiFi.status() != WL_CONNECTED) return;
  String url = String(GOOGLE_LOG_URL) + "?action=log&id=" + String(student->id) +
               "&name=" + urlEncode(student->name) + "&mssv=" + student->mssv +
               "&date=" + currentDate + "&time=" + currentTime +
               "&event=" + urlEncode(action) + "&peopleCount=" + String(peopleCount);
  HTTPClient http;
  WiFiClientSecure client;
  client.setInsecure();
  http.begin(client, url);
  http.setFollowRedirects(HTTPC_FORCE_FOLLOW_REDIRECTS);
  http.GET();
  http.end();
  client.stop();
}

void controlDevicesOnEntry(int oldCount, int newCount) {
  if (digitalRead(SW_LED) == LOW || digitalRead(SW_FAN) == LOW) return;
  if (oldCount == 0 && newCount == 1) {
    digitalWrite(LED, HIGH);
    digitalWrite(FAN, HIGH);
    if (Blynk.connected()) {
      Blynk.virtualWrite(VPIN_LED_CONTROL, 1);
      Blynk.virtualWrite(VPIN_FAN_CONTROL, 1);
    }
  }
}

void controlDevicesOnExit(int oldCount, int newCount) {
  if (digitalRead(SW_LED) == LOW || digitalRead(SW_FAN) == LOW) return;
  if (oldCount == 1 && newCount == 0) {
    digitalWrite(LED, LOW);
    digitalWrite(FAN, LOW);
    if (Blynk.connected()) {
      Blynk.virtualWrite(VPIN_LED_CONTROL, 0);
      Blynk.virtualWrite(VPIN_FAN_CONTROL, 0);
    }
  }
}

void sendStrangerAlert(String reason, String details) {
  String notification = currentTime + " | CẢNH BÁO | " + reason + " | " + details + " | Yêu cầu: Kiểm tra ngay";
  if (Blynk.connected()) {
    Blynk.logEvent("nguoi_la", notification);
    Blynk.virtualWrite(VPIN_LAST_PERSON, notification);
  }
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("  NGUOI LA!     ");
  lcd.setCursor(0, 1);
  lcd.print("  TU CHOI!      ");
  lcdShowingInfo = true;
  lcdDisplayTime = millis() + 3000;
  tick(4, 200);
}

void resetAuthState() {
  authState = AUTH_IDLE;
  authSource = SOURCE_NONE;
  detectedFaceID = 0;
  detectedName = "";
  authStartTime = 0;
}

void processCheckIn(Student* student) {
  tick(1, 60);
  MOCUA;
  doorIsOpen = true;
  doorOpenTime = millis();
  int oldCount = peopleCount;
  totalIn++;
  addPersonToRoom(student->faceID);
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(("MSSV " + student->mssv).substring(0, 16));
  lcd.setCursor(0, 1);
  lcd.print(student->name.substring(0, 16));
  lcdShowingInfo = true;
  lcdDisplayTime = millis() + 3000;
  logToGoogleSheets(student, "VÀO");
  if (Blynk.connected()) {
    String msg = currentTime.substring(0, 5) + " | " + student->name + " (" + student->mssv + ") vào | Phòng: " + String(peopleCount) + " người";
    Blynk.virtualWrite(VPIN_LAST_PERSON, msg);
    Blynk.virtualWrite(VPIN_ROOM_COUNT, peopleCount);
    Blynk.virtualWrite(VPIN_TOTAL_IN, totalIn);
    String notification = currentTime + " | " + student->name + " (MSSV: " + student->mssv + ") | Vào phòng | Số người trong phòng: " + String(peopleCount);
    Blynk.logEvent("nguoi_vao", notification);
  }
  controlDevicesOnEntry(oldCount, peopleCount);
  saveToEEPROM();
}

void processCheckOut(Student* student) {
  if (peopleCount <= 0) {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print(" PHONG TRONG!   ");
    lcd.setCursor(0, 1);
    lcd.print(" KHONG CHO RA!  ");
    lcdShowingInfo = true;
    lcdDisplayTime = millis() + 2000;
    tick(3, 200);
    return;
  }
  if (!isPersonInRoom(student->faceID)) {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print(" CHUA VAO!      ");
    lcd.setCursor(0, 1);
    lcd.print(" KHONG CHO RA!  ");
    lcdShowingInfo = true;
    lcdDisplayTime = millis() + 2000;
    tick(3, 200);
    return;
  }
  tick(1, 60);
  MOCUA;
  doorIsOpen = true;
  doorOpenTime = millis();
  int oldCount = peopleCount;
  totalOut++;
  removePersonFromRoom(student->faceID);
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(("MSSV " + student->mssv).substring(0, 16));
  lcd.setCursor(0, 1);
  lcd.print(student->name.substring(0, 16));
  lcdShowingInfo = true;
  lcdDisplayTime = millis() + 3000;
  logToGoogleSheets(student, "RA");
  if (Blynk.connected()) {
    String msg = currentTime.substring(0, 5) + " | " + student->name + " (" + student->mssv + ") ra | Phòng: " + String(peopleCount) + " người";
    Blynk.virtualWrite(VPIN_LAST_PERSON, msg);
    Blynk.virtualWrite(VPIN_ROOM_COUNT, peopleCount);
    Blynk.virtualWrite(VPIN_TOTAL_OUT, totalOut);
    String notification = currentTime + " | " + student->name + " (MSSV: " + student->mssv + ") | Ra khỏi phòng | Số người trong phòng: " + String(peopleCount);
    Blynk.logEvent("nguoi_ra", notification);
  }
  controlDevicesOnExit(oldCount, peopleCount);
  saveToEEPROM();
}
// NÚT NHẤN 

void handleButtonLED() {
  int reading = digitalRead(SW_LED);
  
  if (reading == LOW) {
    static unsigned long lastPress = 0;
    
    if (millis() - lastPress > 300) {
      lastPress = millis();
      
      int currentState = digitalRead(LED);
      digitalWrite(LED, !currentState);
      
      if (Blynk.connected()) {
        Blynk.virtualWrite(VPIN_LED_CONTROL, !currentState);
      }
      
      tick(1, 50);
      
      Serial.print("LED TOGGLED! State: ");
      Serial.println(!currentState ? "ON" : "OFF");
    }
  }
}

void handleButtonFAN() {
  int reading = digitalRead(SW_FAN);
  
  if (reading == LOW) {
    static unsigned long lastPress = 0;
    
    if (millis() - lastPress > 300) {
      lastPress = millis();
      
      int currentState = digitalRead(FAN);
      digitalWrite(FAN, !currentState);
      
      if (Blynk.connected()) {
        Blynk.virtualWrite(VPIN_FAN_CONTROL, !currentState);
      }
      
      tick(1, 50);
      
      Serial.print("FAN TOGGLED! State: ");
      Serial.println(!currentState ? "ON" : "OFF");
    }
  }
}

BLYNK_WRITE(VPIN_LED_CONTROL) { digitalWrite(LED, param.asInt()); }
BLYNK_WRITE(VPIN_FAN_CONTROL) { digitalWrite(FAN, param.asInt()); }
BLYNK_WRITE(VPIN_SCHEDULE_MODE) { scheduleEnabled = param.asInt(); }
BLYNK_WRITE(VPIN_RELOAD_STUDENTS) {
  if (param.asInt() == 1) {
    loadStudentsFromGoogleSheets();
    Blynk.virtualWrite(VPIN_RELOAD_STUDENTS, 0);
  }
}

BLYNK_WRITE(VPIN_FACE_ID) {
  int id = param.asInt();
  if (id == 0) return;
  int faceID = abs(id);
  
  if (id > 0) {
    Student* student = findStudentByFaceID(faceID);
    if (student == nullptr) {
      sendStrangerAlert("Camera IN", "Khuôn mặt #" + String(faceID) + " không có trong hệ thống");
      resetAuthState();
      return;
    }
    if (isPersonInRoom(faceID)) {
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print(" DA O TRONG!    ");
      lcd.setCursor(0, 1);
      lcd.print(student->name.substring(0, 16));
      lcdShowingInfo = true;
      lcdDisplayTime = millis() + 2000;
      tick(2, 100);
      return;
    }
    if (authState == AUTH_FACE_DETECTED) {
      if (authSource == SOURCE_FACE) {
        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print(" CAN QUET THE!  ");
        lcd.setCursor(0, 1);
        lcd.print(" RFID!          ");
        lcdShowingInfo = true;
        lcdDisplayTime = millis() + 2000;
        tick(3, 150);
        return;
      }
      if (student->faceID == detectedFaceID) {
        processCheckIn(student);
        resetAuthState();
      } else {
        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print(" THE KHONG HOP  ");
        lcd.setCursor(0, 1);
        lcd.print(" LE!            ");
        lcdShowingInfo = true;
        lcdDisplayTime = millis() + 3000;
        tick(4, 200);
        sendStrangerAlert("Thông tin không khớp", "Thẻ không hợp lệ với khuôn mặt");
        resetAuthState();
      }
    } else {
      authState = AUTH_FACE_DETECTED;
      authSource = SOURCE_FACE;
      detectedFaceID = faceID;
      detectedName = student->name;
      authStartTime = millis();
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print(" NHAN DIEN OK!  ");
      lcd.setCursor(0, 1);
      lcd.print(" QUET THE...    ");
      lcdShowingInfo = true;
      lcdDisplayTime = millis() + 20000;
      tick(2, 100);
    }
  } else {
    Student* student = findStudentByFaceID(faceID);
    if (student == nullptr) {
      sendStrangerAlert("Camera OUT", "Khuôn mặt #" + String(faceID) + " không có trong hệ thống");
      return;
    }
    if (!isPersonInRoom(faceID)) {
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print(" CHUA VAO!      ");
      lcd.setCursor(0, 1);
      lcd.print(" KHONG CHO RA!  ");
      lcdShowingInfo = true;
      lcdDisplayTime = millis() + 2000;
      tick(3, 200);
      return;
    }
    if (authState == AUTH_FACE_DETECTED) {
      if (authSource == SOURCE_FACE) {
        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print(" CAN QUET THE!  ");
        lcd.setCursor(0, 1);
        lcd.print(" RFID!          ");
        lcdShowingInfo = true;
        lcdDisplayTime = millis() + 2000;
        tick(3, 150);
        return;
      }
      if (student->faceID == detectedFaceID) {
        processCheckOut(student);
        resetAuthState();
      } else {
        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print(" THE KHONG HOP  ");
        lcd.setCursor(0, 1);
        lcd.print(" LE!            ");
        lcdShowingInfo = true;
        lcdDisplayTime = millis() + 3000;
        tick(4, 200);
        sendStrangerAlert("Thông tin không khớp", "Thẻ không hợp lệ với khuôn mặt");
        resetAuthState();
      }
    } else {
      authState = AUTH_FACE_DETECTED;
      authSource = SOURCE_FACE;
      detectedFaceID = faceID;
      detectedName = student->name;
      authStartTime = millis();
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print(" NHAN DIEN OK!  ");
      lcd.setCursor(0, 1);
      lcd.print(" QUET THE RA... ");
      lcdShowingInfo = true;
      lcdDisplayTime = millis() + 20000;
      tick(2, 100);
    }
  }
}

BLYNK_WRITE(VPIN_SCHEDULE_ON) {
  TimeInputParam t(param);
  if (t.hasStartTime()) {
    scheduleON_hour = t.getStartHour();
    scheduleON_min = t.getStartMinute();
  }
}

BLYNK_WRITE(VPIN_SCHEDULE_OFF) {
  TimeInputParam t(param);
  if (t.hasStartTime()) {
    scheduleOFF_hour = t.getStartHour();
    scheduleOFF_min = t.getStartMinute();
  }
}

BLYNK_CONNECTED() {
  Blynk.virtualWrite(VPIN_ROOM_COUNT, peopleCount);
  Blynk.virtualWrite(VPIN_TOTAL_IN, totalIn);
  Blynk.virtualWrite(VPIN_TOTAL_OUT, totalOut);
}

void updateBlynk() {
  if (Blynk.connected()) Blynk.virtualWrite(VPIN_ROOM_COUNT, peopleCount);
}

void scheduleControl() {
  if (!scheduleEnabled) return;
  if (hh == scheduleON_hour && mm == scheduleON_min && !scheduleTurnedOn) {
    digitalWrite(LED, HIGH);
    digitalWrite(FAN, HIGH);
    Blynk.virtualWrite(VPIN_LED_CONTROL, 1);
    Blynk.virtualWrite(VPIN_FAN_CONTROL, 1);
    scheduleTurnedOn = true;
    scheduleTurnedOff = false;
  }
  if (hh == scheduleOFF_hour && mm == scheduleOFF_min && !scheduleTurnedOff) {
    digitalWrite(LED, LOW);
    digitalWrite(FAN, LOW);
    Blynk.virtualWrite(VPIN_LED_CONTROL, 0);
    Blynk.virtualWrite(VPIN_FAN_CONTROL, 0);
    scheduleTurnedOff = true;
    scheduleTurnedOn = false;
  }
  if (mm != scheduleON_min) scheduleTurnedOn = false;
  if (mm != scheduleOFF_min) scheduleTurnedOff = false;
}

void checkWiFiConnection() {
  if (millis() - lastWiFiCheck < 30000) return;
  lastWiFiCheck = millis();
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.disconnect();
    delay(1000);
    WiFi.begin(ssid, pass);
    int retries = 0;
    while (WiFi.status() != WL_CONNECTED && retries < 20) {
      delay(500);
      retries++;
    }
  }
}

void checkBlynkConnection() {
  if (millis() - lastBlynkCheck < 60000) return;
  lastBlynkCheck = millis();
  if (!Blynk.connected() && WiFi.status() == WL_CONNECTED) {
    Blynk.disconnect();
    delay(1000);
    Blynk.connect(5000);
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  pinInit();
  lcd.setCursor(0, 0);
  lcd.print(" Connecting ... ");
  EEPROM.begin(EEPROM_SIZE);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, pass);
  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 40) {
    delay(500);
    retries++;
  }
  Blynk.config(BLYNK_AUTH_TOKEN);
  if (Blynk.connect(10000)) {
    lcd.setCursor(0, 1);
    lcd.print("  Connected!!!  ");
  }
  timeClient.begin();
  timeClient.setUpdateInterval(3600000);
  updateTime();
  loadFromEEPROM();
  
  // KHÔI PHỤC TRẠNG THÁI ĐÈN/QUẠT SAU MẤT ĐIỆN (AN TOÀN)
  
  // Đếm ngược 5 giây
  for (int i = 5; i > 0; i--) {
    Serial.print(".");
    delay(1000);
  }
  
  // Kiểm tra có người trong phòng không
  if (peopleCount > 0) {
   
    // Kiểm tra switch tự động (nếu OFF thì không bật)
    bool ledAutoEnabled = (digitalRead(SW_LED) == HIGH);
    bool fanAutoEnabled = (digitalRead(SW_FAN) == HIGH);
    
    if (ledAutoEnabled) {
      digitalWrite(LED, HIGH);
      
      // Cập nhật Blynk
      if (Blynk.connected()) {
        Blynk.virtualWrite(VPIN_LED_CONTROL, 1);
      }
    } else {
      Serial.println("LED: Switch OFF, khong bat");
    }
    
    if (fanAutoEnabled) {
      digitalWrite(FAN, HIGH);
      
      // Cập nhật Blynk
      if (Blynk.connected()) {
        Blynk.virtualWrite(VPIN_FAN_CONTROL, 1);
      }
    } else {
      Serial.println("FAN: Switch OFF, khong bat");
    }
    
    // Hiển thị LCD
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("PHONG CO ");
    lcd.print(peopleCount);
    lcd.print(" NGUOI");
    lcd.setCursor(0, 1);
    lcd.print("DA BAT LAI!     ");
    delay(3000);
  } else {
     
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("PHONG TRONG     ");
    lcd.setCursor(0, 1);
    lcd.print("KHONG CO NGUOI       ");
    delay(2000);
  }
  
  lcd.clear();
  // ═══════════════════════════════════════════════════════════════════════════
  
  loadStudentsFromGoogleSheets();
  timer.setInterval(10000L, updateBlynk);
  timer.setInterval(120000L, saveToEEPROM);
  tick(2, 100);
  delay(1500);
  lcd.clear();

}

void loop() {
  checkWiFiConnection();
  checkBlynkConnection();
  if (Blynk.connected()) Blynk.run();
  timer.run();
  
  // Xử lý nút nhấn
  handleButtonLED();
  handleButtonFAN();
  
  if (millis() - lastTimeUpdate > 1000) {
    lastTimeUpdate = millis();
    updateTime();
    scheduleControl();
  }
  
  if (!lcdShowingInfo) {
    lcd.setCursor(0, 0);
    lcd.print("PHONG THI NGHIEM");
    lcd.setCursor(0, 1);
    lcd.print(" ");
    lcd.print(currentTime);
    lcd.print(" SL:");
    lcd.print(peopleCount);
    lcd.print("  ");
  }
  
  if (lcdShowingInfo && (millis() >= lcdDisplayTime)) {
    lcdShowingInfo = false;
    lcd.clear();
  }
  
  if (authState == AUTH_FACE_DETECTED && (millis() - authStartTime > AUTH_TIMEOUT)) {
    resetAuthState();
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print(" TIMEOUT!       ");
    lcd.setCursor(0, 1);
    lcd.print(" THU LAI...     ");
    delay(1000);
    lcd.clear();
  }
  
  if (mfrc522.PICC_IsNewCardPresent() && mfrc522.PICC_ReadCardSerial()) {
    tick(1, 60);
    String rfid = rfidToString(mfrc522.uid.uidByte, mfrc522.uid.size);
    Student* studentByRFID = findStudentByRFID(rfid);
    
    if (studentByRFID == nullptr) {
      sendStrangerAlert("RFID không hợp lệ", "RFID: " + rfid);
      resetAuthState();
    } else if (authState == AUTH_FACE_DETECTED) {
      if (authSource == SOURCE_RFID) {
        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print(" CAN NHAN DIEN  ");
        lcd.setCursor(0, 1);
        lcd.print(" KHUON MAT!     ");
        lcdShowingInfo = true;
        lcdDisplayTime = millis() + 2000;
        tick(3, 150);
        mfrc522.PICC_HaltA();
        mfrc522.PCD_StopCrypto1();
        return;
      }
      if (studentByRFID->faceID == detectedFaceID) {
        if (isPersonInRoom(studentByRFID->faceID)) processCheckOut(studentByRFID);
        else processCheckIn(studentByRFID);
        resetAuthState();
      } else {
        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print(" KHUON MAT KHAC ");
        lcd.setCursor(0, 1);
        lcd.print(" VOI THE!       ");
        lcdShowingInfo = true;
        lcdDisplayTime = millis() + 3000;
        tick(4, 200);
        sendStrangerAlert("Thông tin không khớp", "Khuôn mặt khác với thẻ RFID");
        resetAuthState();
      }
    } else {
      authState = AUTH_FACE_DETECTED;
      authSource = SOURCE_RFID;
      detectedFaceID = studentByRFID->faceID;
      detectedName = studentByRFID->name;
      authStartTime = millis();
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print(" DA QUET THE!   ");
      lcd.setCursor(0, 1);
      lcd.print(" NHAN DIEN MAT...");
      lcdShowingInfo = true;
      lcdDisplayTime = millis() + 20000;
      tick(2, 100);
    }
    mfrc522.PICC_HaltA();
    mfrc522.PCD_StopCrypto1();
  }
  
  if (doorIsOpen && (millis() - doorOpenTime >= 1000)) {
    DONGCUA;
    doorIsOpen = false;
  }
  delay(50);
}

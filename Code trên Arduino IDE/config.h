#define BUZZ        13
#define LED         4
#define FAN         2
#define SW_LED      36
#define SW_FAN      39

#include <Wire.h> 
#include <LiquidCrystal_I2C.h>
LiquidCrystal_I2C lcd(0x27, 16, 2);

#include <SPI.h>
#include <MFRC522.h>
MFRC522 mfrc522(5, 255); 

#include <ESP32Servo.h>
Servo myservo;
#define MOCUA   myservo.write(100);
#define DONGCUA myservo.write(200);

void tick(int x=1, int y=90)
{
  while(x--)
  {
    digitalWrite(BUZZ,1); delay(y);
    digitalWrite(BUZZ,0); delay(y);
  }
}
void pinInit()
{
  pinMode(SW_LED,INPUT); pinMode(SW_FAN,INPUT);
  pinMode(LED,OUTPUT); digitalWrite(LED,0);
  pinMode(FAN,OUTPUT); digitalWrite(FAN,0);
  pinMode(BUZZ,OUTPUT); tick();
  
  lcd.init();
  lcd.backlight();

  ESP32PWM::allocateTimer(0);
	ESP32PWM::allocateTimer(1);
	ESP32PWM::allocateTimer(2);
	ESP32PWM::allocateTimer(3);
	myservo.setPeriodHertz(50); myservo.attach(15, 500, 2500);
  DONGCUA;

	SPI.begin();
	mfrc522.PCD_Init();
}



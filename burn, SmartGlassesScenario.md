Project: SmartGlasses  
AppName: OOOO  
LastUpdate: 2017.12.27 15:53 최환영 (GitHub상에서 원본 수정 금지)  
Version: beta 1.0  

*주요기술: Bluetooth, NFC, SQLite, TTS  
*Platform: Android  
*Log: 2017.12.27 Project Scenario beta  
      2017.12.31 App UI Wireframe beta

#App 실행 방법  
-NFC(혹은 다른 기술)이용, Tag시 App 실행(시각 장애인을 위한 실행방법)  
-직접 실행  

#App 처음 실행 시  
-MainView가 나타남  
-App 처음 실행 시 초기 Setting이 필요(보호자가 이용하게 될 것)  
-초기 Setting을 해두면 다음부터는 건드릴 필요 없음(보호자 없어도 됨)  

#Main  
Main에서 화면을 tap하여 Setting에서 등록한 기기와 연결한다.  

#Setting   
-Setting 정보는 App종료 후에도 유지되어야 한다.  
-Setting 정보는 target device와 Bluetooth connection 정보를 유지한다.  

#UI설계  
1)Splash  

2)MainView  
-시각장애인이 이용하므로 간단한 UI로 설계한다.  
-TTS이용, target device와 연결 시도 중이라고 알려준다.  
-연결 완료 시 tap하여 연결하도록 유도한다.  
-Display 임의의 곳을 tap하면 연결되도록 한다.  
-Main화면에서 SettingView로 가기 위한 button이 존재한다.  
-그 button은 right|bottom에 원형으로 존재한다.  
-button 터치시 MainView를 덮는 형태로 SettingView가 나타난다.  
 
3)SettingView  
-Bluetooth connection을 최초 수동으로 설정한다.  
-연결정보는 저장되어 앱 재실행 해도 저장된 연결만 한다.  

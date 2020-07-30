using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class Carrot : Agent
{
    //Prototype Setting
    //최적 : 18C˚ >> [-30 ~ 30]
    //최적 : 토양 표면이 마르는 시기의 7 ~ 10일 간격 >> 500ml정도 추측이나 급수량은 추후 보정
    //조생 : 70 ~ 80(일)  중생 : 90 ~ 100(일)   만생 : 120일 이상(국내 조생종 다수)
    //생장 : False :: 수확 : True ++ 카메라로 줄기지면접촉 논의
    //정상 : True :: 비정상 : False >> 병충해등의 이유로 잎 색깔이 정상 범주 벗어날 시, 메세지
    //[노란색 = 무름병 >> 해결책 : 토양산도 상승 및 약제 공급 행동]  [검은색 = 검은 잎마름병 >> 해결책 : 수분 및 약제 공급 행동]
    //최적 : 6pH >> [0 ~ 14]

    //public double Acidity = 0;            
    //Pesticide
    //Nutrients

    public double Temp = 0;
    public int Humid_cycle = 0;
    //public int Harv_lim = 0;
    public bool Normal = true;
    //public double pH = 0;

    // Start is called before the first frame update
    void Start()
    {
        Temp = 18.0;
        Humid_cycle = 8;
    }

    // Update is called once per frame
    void Update()
    {

    }
}

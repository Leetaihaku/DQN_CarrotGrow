using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class Capsule : Agent
{
    Rigidbody RGCapsule;
    public Transform TRCapsule;
    //void Start()
    //{
    //    RGCapsule = GetComponent<Rigidbody>();
    //}

    public Transform Portal_L;

    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            // If the Agent fell, zero its momentum
            this.RGCapsule.angularVelocity = Vector3.zero;
            this.RGCapsule.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        TRCapsule.localPosition = new Vector3(0.17f, 0.65f, -0.9f);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        //Portal & Agent Position
        sensor.AddObservation(Portal_L.localPosition);
        //sensor.AddObservation(Portal_R.localPosition);
        sensor.AddObservation(this.transform.localPosition);

        // Agent velocity
        sensor.AddObservation(RGCapsule.velocity.x);
        sensor.AddObservation(RGCapsule.velocity.z);

        //Carrot Status
        //Carrot Carrot = GameObject.Find("Carrot").GetComponent<Carrot>();
        //Carrot

    }

    public float forceMultiplier = 10;

    public override void OnActionReceived(float[] vectorAction)
    {
        // Actions, size = 2
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = vectorAction[0];
        controlSignal.z = vectorAction[1];
        RGCapsule.AddForce(controlSignal * forceMultiplier);

        float distance = Vector3.Distance(this.transform.localPosition, Portal_L.localPosition);

        // Rewards
        //정상행동 선택 >> 보상수여 + END
        if (distance < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }

        //비정상행동 선택 >> 패널티 + END
        if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }

        /*
        // Reached target
        if (distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }

        // Fell off platform
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
        */
    }

}

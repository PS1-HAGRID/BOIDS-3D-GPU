using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

public struct Boid
{
    public float3 position;
    public float3 velocity;
    public quaternion rotation;
}
public struct MatrixHelper
{
    public static Matrix4x4 MatrixBuilder(float3 position, Quaternion rotation, Vector3 scale)
    {
        return Matrix4x4.TRS(position, rotation, scale);
    }
}

public class BoidsSimCPUGPU : MonoBehaviour
{
    [Header("Simulation Parameters")]
    [SerializeField] int _NumBoids;
    [SerializeField] BoxCollider _Bounds;

    [Header("Boid Parameters")]
    [SerializeField] float _FieldOfViewRadius = 10;
    [SerializeField] float _ProtectedRangeRadius = 3;
    [SerializeField] float _MatchingFactor = 0.05f;
    [SerializeField] float _CohesionFactor = 0.0005f;
    [SerializeField] float _AvoidanceFactor = 0.05f;
    [SerializeField] float _BoundsAvoidanceForce = 0.5f;
    [SerializeField] float _Speed = 1;
    [SerializeField] float3 _BoidsScale = new float3(1,1,1);

    [Header("Renderer Parameters")]
    [SerializeField] Mesh _BoidMesh;
    [SerializeField] Material _BoidMaterial;
    [Range(0,1000)] [SerializeField] int _NumberOfBoidsPerBatches;

    List<Boid> _Boids = new List<Boid>();

    List<List<Matrix4x4>> _Batches = new List<List<Matrix4x4>>(); 

    public float3 RandomDirection()
    {
        float3 lDir = new float3(UnityEngine.Random.Range(-1,1), UnityEngine.Random.Range(-1,1), UnityEngine.Random.Range(-1,1));

        return lDir;
    }

    public void Init()
    {
        _Batches.Add(new List<Matrix4x4>());
        int lCurrentBatch = 0;
        int lBoidCounter = 0;

        for (int lCurrentBoid = 0; lCurrentBoid < _NumBoids; lCurrentBoid++) 
        {
            Boid lBoid = new Boid();
            lBoid.position = RandomDirection() * 2;
            lBoid.velocity = RandomDirection();
            lBoid.rotation = quaternion.Euler(new float3());

            _Boids.Add(lBoid);

            if (lBoidCounter == _NumberOfBoidsPerBatches)
            {
                _Batches.Add(new List<Matrix4x4>());
                lBoidCounter = 0;
                lCurrentBatch++;
            }

            _Batches[lCurrentBatch].Add(MatrixHelper.MatrixBuilder(lBoid.position,lBoid.rotation,_BoidsScale));

            lBoidCounter++;
        }

        Debug.Log(_Batches.Count);
        Debug.Log(_Batches[0].Count);
    }

    public void SimStep()
    {
        float lSquaredProtectedRange = _ProtectedRangeRadius * _ProtectedRangeRadius;
        float lSquaredFieldOfView = _FieldOfViewRadius * _FieldOfViewRadius;

        int lBoidIndex = 0;
        int lCurrentBatch = 0;

        for(int lCurrentBoid = 0;lCurrentBoid < _NumBoids; lCurrentBoid++) 
        {
           
            int NumNeighbors = 0;

            Boid currentBoid = _Boids[lCurrentBoid];

            float3 matching = new float3();
            float3 lCohesion = new float3();
            float3 avoidance = new float3();
            
            for(int lOtherBoid = 0; lOtherBoid < _Batches.Count; lOtherBoid++)
            {
                if(CompareVectors(currentBoid.position, _Boids[lOtherBoid].position))
                {
                    continue;
                }

                float lSquaredDistanceBetweenBoids = (_Boids[lOtherBoid].position.x - currentBoid.position.x) * (_Boids[lOtherBoid].position.x - currentBoid.position.x) + (_Boids[lOtherBoid].position.y - currentBoid.position.y) * (_Boids[lOtherBoid].position.y - currentBoid.position.y) + (_Boids[lOtherBoid].position.z - currentBoid.position.z) * (_Boids[lOtherBoid].position.z - currentBoid.position.z);

                if(lSquaredDistanceBetweenBoids > _FieldOfViewRadius)
                {
                    continue;
                }

                NumNeighbors++;
                lCohesion += _Boids[lOtherBoid].position;
                matching += _Boids[lOtherBoid].velocity;

                if(lSquaredDistanceBetweenBoids < lSquaredProtectedRange)
                {
                    avoidance += currentBoid.position - _Boids[lOtherBoid].position;
                }
            }

            if(NumNeighbors != 0)
            {
                lCohesion /= NumNeighbors;
                matching /= NumNeighbors;
            }

            currentBoid.velocity += ((lCohesion - currentBoid.position) * _CohesionFactor ) + ((matching - currentBoid.velocity) * _MatchingFactor) + (avoidance * _AvoidanceFactor);

            currentBoid.velocity += CheckBounds(currentBoid.position) * _BoundsAvoidanceForce;

            currentBoid.velocity = math.normalizesafe(currentBoid.velocity);

            //currentBoid.rotation = quaternion.LookRotation(currentBoid.velocity, new float3(0, 0, 1));

            currentBoid.velocity *= _Speed;

            currentBoid.position += currentBoid.velocity * Time.deltaTime;

            _Boids[lCurrentBoid] = currentBoid;

            //Preparing Batches
            if(lBoidIndex == _NumberOfBoidsPerBatches)
            {
                lCurrentBatch++;
                lBoidIndex = 0;
            }

            _Batches[lCurrentBatch][lBoidIndex] = MatrixHelper.MatrixBuilder(_Boids[lCurrentBoid].position, _Boids[lCurrentBoid].rotation, _BoidsScale);

            lBoidIndex++;
        }
    }
    float3 CheckBounds(float3 position)
    {
        float3 vel = new float3(0, 0, 0);
        vel.x = (position.x < _Bounds.center.x - _Bounds.size.x * 0.5f) ? 1.0f : ((position.x > _Bounds.center.x + _Bounds.size.x * 0.5f) ? -1.0f : 0f);
        vel.y = (position.y < _Bounds.center.y - _Bounds.size.y * 0.5f) ? 1.0f : ((position.y > _Bounds.center.y + _Bounds.size.y * 0.5f) ? -1.0f : 0f);
        vel.z = (position.z < _Bounds.center.z - _Bounds.size.z * 0.5f) ? 1.0f : ((position.z > _Bounds.center.z + _Bounds.size.z * 0.5f) ? -1.0f : 0f);

        return vel;
    }

    public bool CompareVectors(float3 vectorA, float3 vectorB)
    {
        return vectorA.Equals(vectorB);
    }


    public void Render()
    {
        for(int lBatchIndex = 0; lBatchIndex < _Batches.Count; lBatchIndex++)
        {
            Graphics.DrawMeshInstanced(_BoidMesh, 0, _BoidMaterial, _Batches[lBatchIndex]);
        }
    }



    void Start()
    {
        Init();
    }

    void Update()
    {
        SimStep();
        Render();
    }
}

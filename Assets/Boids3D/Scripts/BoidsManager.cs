using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Matrix4x4 = UnityEngine.Matrix4x4;
using Quaternion = UnityEngine.Quaternion;
using Random = UnityEngine.Random;
using Vector3 = UnityEngine.Vector3;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Transforms;
using UnityEngine.Rendering;
using Unity.Entities.UniversalDelegates;

public struct boidData
{
    public float3 position;
    public float3 velocity;
}

public class BoidsManager : MonoBehaviour
{
    [Header("simulation params")]
    [Range(128, 100000)]
    public int numBoids = 20000;
    public BoxCollider bounds;
    Vector4 boundsSize;
    Vector4 boundsCenter;

    [Header("boids params")]
    public float matchingRadius = 1f;
    public float matchingFactor = 0.5f;
    public int maxNeighbors = 20;
    public float centeringRadius = 1f;
    public float centeringFactor = 0.5f;
    public float avoidanceRadius = 0.5f;
    public float avoidanceFactor = 2f;
    public float steerAwayFromBoundsForce = 10f;
    public float maxSpeed = 4, maxSteeringSpeed = 10;

    private Quaternion[] boidsRotations;
    public Vector3 boidsScale = new Vector3(1,1,1);
    private List<List<Matrix4x4>> batches = new List<List<Matrix4x4>>();

    [Header("Gpu Instancing vars")]
    public Material boidMaterial;
    public Mesh boidMesh;

    [Header("Compute Shader vars")]
    public ComputeShader boidGPUPhysicsComputeShader;
    private const string BOIDS_DATA_KERNEL_NAME = "BoidsDataCalcs"; 
    private const string BOIDS_STEERING_FORCES_KERNEL_NAME = "BoidForceCalcs";
    private int boidsDataKernelIndex, boidsSteeringForceKernelIndex, dispatchedThreadGroupSize;
    private uint storedThreadGroupSize;
    private ComputeBuffer boidsDataBuffer;
    private ComputeBuffer boidsSteeringForcesBuffer;
    int boidsDataBufferStride = sizeof(float) * 6;
    //int boidSteeringForcesBufferStride = sizeof(float) * 3;
    boidData[] boidDatas;
    float3[] boidForces;
    private Vector3 PositionBoidRandomly()
    {
        //Creates a new random Vector3
        return new Vector3(Random.Range(0, 50), Random.Range(0, 50), Random.Range(0, 50));
    }

    private Vector3 GiveRandomVelocity()
    {
        return new Vector3(Random.Range(0f, 1f), Random.Range(0f, 1f), Random.Range(0f, 1f));
    }

    //Gpu instancing
    private void RenderBoids()
    {
        for (int i = 0; i < batches.Count; i++)
        {
            Graphics.DrawMeshInstanced(boidMesh, 0, boidMaterial, batches[i]);
        }

    }

    //returns a quaternion pointiong to vector3 direction dir
    private Quaternion PointToDirection(Vector3 pDir)
    {
        return Quaternion.FromToRotation(Vector3.up, pDir);
    }

    private void ReleaseBuffers()
    {
        ReleaseBufferSafe(ref boidsDataBuffer);
        ReleaseBufferSafe(ref boidsSteeringForcesBuffer);
    }

    //Releases and nullifies a Compute Buffer
    private void ReleaseBufferSafe(ref ComputeBuffer pBuffer)
    {
        if (pBuffer == null) return;
        pBuffer.Release();
        pBuffer = null;
    }


    void Init()
    {
        //Bounds Init
        boundsCenter = bounds.center;
        boundsSize = bounds.size;

        //local var to prepare GPU rendering
        int lCurrentBatch = 0, lBoidCounter = 0;

        //Kernels init
        //boidsDataKernelIndex = boidGPUPhysicsComputeShader.FindKernel("BoidsDataCalcs");
        boidsSteeringForceKernelIndex = boidGPUPhysicsComputeShader.FindKernel(BOIDS_STEERING_FORCES_KERNEL_NAME);

        //Buffers init
        boidsDataBuffer = new ComputeBuffer(numBoids, boidsDataBufferStride);
        //boidsSteeringForcesBuffer = new ComputeBuffer(numBoids,boidSteeringForcesBufferStride);

        //Batches init
        batches.Add(new List<Matrix4x4>());

        //Boid arrays init
        boidDatas = new boidData[numBoids];
        boidForces = new float3[numBoids];
        boidsRotations = new Quaternion[numBoids];
        for (int lCurrentBoid = 0; lCurrentBoid < numBoids; lCurrentBoid++) 
        {
            //arrays population
            boidDatas[lCurrentBoid].position = PositionBoidRandomly();
            boidDatas[lCurrentBoid].velocity = GiveRandomVelocity();
            boidForces[lCurrentBoid] = new float3();
            boidsRotations[lCurrentBoid] = Quaternion.identity;

            //Populating Batches for GPU rendering
            batches[lCurrentBatch].Add(MatrixHelper.MatrixBuilder(new float3(), Quaternion.identity, new Vector3()));
            
            lBoidCounter++;

            if (lBoidCounter > 998) 
            {
                batches.Add(new List<Matrix4x4>());
                lBoidCounter = 0;
                lCurrentBatch++;
            }

        }

        //Buffers data assignment
        boidsDataBuffer.SetData(boidDatas);
        //boidsSteeringForcesBuffer.SetData(boidForces);

        //Preparing thread groups sizes
        boidGPUPhysicsComputeShader.GetKernelThreadGroupSizes(boidsSteeringForceKernelIndex, out storedThreadGroupSize, out _, out _);
        
        dispatchedThreadGroupSize = numBoids / (int)storedThreadGroupSize;

        if (dispatchedThreadGroupSize % storedThreadGroupSize == 0) return;

        while (dispatchedThreadGroupSize % storedThreadGroupSize != 0)
        {
            dispatchedThreadGroupSize += 1;
            if (dispatchedThreadGroupSize % storedThreadGroupSize != 0) continue;
        }
    }

    private void BoidSimStep()
    {
        //Compute shader check
        if (boidGPUPhysicsComputeShader == null)
        {
            throw new Exception("No Compute Shader Assigned");
        }

        //Connecting Buffers
        boidGPUPhysicsComputeShader.SetBuffer(boidsSteeringForceKernelIndex, "boidDatasBuffer", boidsDataBuffer);

        //Assigning variables
        boidGPUPhysicsComputeShader.SetInt("numBoids", numBoids);

        boidGPUPhysicsComputeShader.SetFloat("matchingRadius", matchingRadius);
        boidGPUPhysicsComputeShader.SetFloat("matchingFactor", matchingFactor);
        boidGPUPhysicsComputeShader.SetFloat("centeringRadius", centeringRadius);
        boidGPUPhysicsComputeShader.SetFloat("centeringFactor", centeringFactor);
        boidGPUPhysicsComputeShader.SetFloat("avoidanceRadius", avoidanceRadius);
        boidGPUPhysicsComputeShader.SetFloat("avoidanceFactor", avoidanceFactor);
        boidGPUPhysicsComputeShader.SetFloat("steerAwayFromBoundForce", steerAwayFromBoundsForce);
        boidGPUPhysicsComputeShader.SetFloat("maxSpeed", maxSpeed);
        boidGPUPhysicsComputeShader.SetFloat("maxSteeringSpeed", maxSteeringSpeed);
        boidGPUPhysicsComputeShader.SetFloat("deltaTime", Time.deltaTime);

        boidGPUPhysicsComputeShader.SetVector("boundsSize", boundsSize);
        boidGPUPhysicsComputeShader.SetVector("boundsCenter", boundsCenter);

        //Dispatching Kernels
        boidGPUPhysicsComputeShader.Dispatch(boidsSteeringForceKernelIndex, dispatchedThreadGroupSize, 1, 1);

        //Interpreting results
        RenderBoids();
        CopyBuffer();
    }

    private void BoidSimCPU()
    {
        for (int currentBoid = 0; currentBoid < boidDatas.Length; currentBoid++)
        {

            float3 velocity = new float3();
            float3 matchingVector = new float3();
            float3 avoidanceVector = new float3();
            float3 centeringVector = new float3();

            int matchingNeighbors = 0;
            int centeringNeighbors = 0;
            int closeNeighbors = 0;


            foreach (boidData otherBoid in boidDatas)
            {
                if (otherBoid.position.x == boidDatas[currentBoid].position.x && otherBoid.position.z == boidDatas[currentBoid].position.z && otherBoid.position.y == boidDatas[currentBoid].position.y)
                {
                    continue;
                }

                float distance = Mathf.Abs((boidDatas[currentBoid].position.x - otherBoid.position.x) + (boidDatas[currentBoid].position.y - otherBoid.position.y) + (boidDatas[currentBoid].position.z - otherBoid.position.z));
                distance *= distance;

                if (distance < matchingRadius * matchingRadius)
                {
                    matchingVector += otherBoid.velocity;
                    matchingNeighbors++;
                }

                if (distance < avoidanceRadius * avoidanceRadius)
                {
                    avoidanceVector += new float3(boidDatas[currentBoid].position.x - otherBoid.position.x, boidDatas[currentBoid].position.y - otherBoid.position.y, boidDatas[currentBoid].position.z - otherBoid.position.z);
                    closeNeighbors++;
                }

                if (distance < centeringRadius * centeringRadius)
                {
                    centeringVector += otherBoid.position;
                    centeringNeighbors++;
                }

            }

            if (matchingNeighbors > 0)
            {
                matchingVector = new float3(matchingVector.x / matchingNeighbors, matchingVector.y / matchingNeighbors, matchingVector.z / matchingNeighbors);
            }

            if (centeringNeighbors > 0)
            {
                centeringVector = new float3(centeringVector.x / centeringNeighbors, centeringVector.y / centeringNeighbors, centeringVector.z / centeringNeighbors);
            }

            velocity = (velocity + (centeringVector - boidDatas[currentBoid].position) * centeringFactor + (matchingVector - boidDatas[currentBoid].velocity) * matchingFactor);
            velocity += avoidanceVector * avoidanceFactor;

            velocity = Limit(velocity, maxSpeed);
            velocity += CheckBounds(boidDatas[currentBoid].position) * steerAwayFromBoundsForce;
            boidDatas[currentBoid].velocity = velocity;
            boidDatas[currentBoid].position += boidDatas[currentBoid].velocity * Time.deltaTime;
        }

        CopyArray();
        RenderBoids();
    }

    float3 Limit(float3 vec, float maximum)
    {
        float3 result = new float3(Math.Clamp(vec.x, 0, maxSpeed),Math.Clamp(vec.y, 0, maxSpeed),Math.Clamp(vec.z, 0, maxSpeed));
        
        return result;
    }

    float3 CheckBounds(float3 position)
    {
        float3 vel = new float3(0, 0, 0);
        vel.x = (position.x < boundsCenter.x - boundsSize.x * 0.5f) ? 1.0f : ((position.x > boundsCenter.x + boundsSize.x * 0.5f) ? -1.0f : 0f);
        vel.y = (position.y < boundsCenter.y - boundsSize.y * 0.5f) ? 1.0f : ((position.y > boundsCenter.y + boundsSize.y * 0.5f) ? -1.0f : 0f);
        vel.z = (position.z < boundsCenter.z - boundsSize.z * 0.5f) ? 1.0f : ((position.z > boundsCenter.z + boundsSize.z * 0.5f) ? -1.0f : 0f);

        return vel;
    }

    void CopyArray()
    {
        int currentBatch = 0, boidCount = 0;
        for (int currentBoid = 0; currentBoid < numBoids; currentBoid++)
        {
            boidsRotations[currentBoid] = PointToDirection(boidDatas[currentBoid].velocity);
            batches[currentBatch][boidCount] = MatrixHelper.MatrixBuilder(boidDatas[currentBoid].position, boidsRotations[currentBoid], boidsScale);
            boidCount++;

            if (boidCount > 998)
            {
                currentBatch++;
                boidCount = 0;
            }

        }

    }


    //Adds every item of a buffer to the batches
    void CopyBuffer()
    {
        boidsDataBuffer.GetData(boidDatas);
        int currentBatch = 0, boidCount = 0;
        for (int currentBoid = 0; currentBoid < numBoids; currentBoid++) 
        {
            boidsRotations[currentBoid] = PointToDirection(boidDatas[currentBoid].velocity);
            batches[currentBatch][boidCount] = MatrixHelper.MatrixBuilder(boidDatas[currentBoid].position, boidsRotations[currentBoid], boidsScale);
            boidCount++;

            if (boidCount > 998)
            {
                currentBatch++;
                boidCount = 0;
            }

        }
        
    }
    void Start()
    {
        Init();
    }

    void Update()
    {
        BoidSimCPU();
    }

    private void OnDestroy()
    {
        ReleaseBuffers();
    }
}



using System;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Mathematics;
using UnityEngine;
using Random = UnityEngine.Random;

public struct BoidData
{
    public float3 Position;
    public float3 Velocity;
    public int Group;
}

public class BoidsDefinitiveScript : MonoBehaviour
{
    [Header("boids params")]
    [Range(1, 100000)]
    public int numberOfBoids;
    public float fieldOfView;
    public float protectedRange;
    public float maxSpeed;
    public Mesh boidMesh;
    public Material boidMaterial;
    public Vector3 boidScale;
    public float centeringFactor = 0.0005f;
    public float alignmentFactor = 0.05f;
    public float avoidanceFactor = 0.05f;
    public float steerAwayFromBoundsStrenght = 0.2f;
    public int numberOfGroups = 1;

    [Header("sim params")]
    [SerializeField] private ComputeShader _BoidsLogicShader;
    [SerializeField] BoxCollider _Bounds;
    [SerializeField] private int _MaxNeightbors;
    [SerializeField] private Camera _Camera;

    const int THREADS_PER_GROUP = 256;
    const int ITEMS_PER_BATCHES = 1000;
    int _NumThreadGroupsForBoidsToDispatch = 1;
    int BoidsLogicKernelIndex = 0;
    int SpatialHashingKernelIndex = 1;

    BoidData[] boidDatas;
    Matrix4x4[] boidMatrices;

    List<Matrix4x4[]> batches = new List<Matrix4x4[]>();

    ComputeBuffer _InputBuffer;
    ComputeBuffer _OutputBuffer;

    ComputeBuffer _HashTableBuffer;
    ComputeBuffer _HashStartBuffer;
    ComputeBuffer _QueryIDs;
    ComputeBuffer _EmptyQueryBuffer;

    int _CellsPerX;
    int _CellsPerY;
    int _CellsPerZ;
    int _TotalCells;
    float _CellSize;
    int hashTableSize;

    SpatialHash hashTable;

    private int AssignGroup(float3 position, float3 centerOfSimulation)
    {
        int group = 0;

        if(position.y >= centerOfSimulation.y)
        {
            if(position.x >= centerOfSimulation.x)
            {
                if(position.z >= centerOfSimulation.z)
                {
                    group = 8;
                }
                else
                {
                    group = 7;
                }
            }
            else
            {
                if (position.z >= centerOfSimulation.z)
                {
                    group = 6;
                }
                else
                {
                    group = 5;
                }
            }
        }
        else
        {
            if (position.x >= centerOfSimulation.x)
            {
                if (position.z >= centerOfSimulation.z)
                {
                    group = 4;
                }
                else
                {
                    group = 3;
                }
            }
            else
            {
                if (position.z >= centerOfSimulation.z)
                {
                    group = 2;
                }
                else
                {
                    group = 1;
                }
            }
        }


        return group;
    }

    private void Init()
    {
        hashTableSize = numberOfBoids * 2;

        //initializing values for spatial hash grid
        float _BBvolume = _Bounds.bounds.size.x * _Bounds.bounds.size.y * _Bounds.bounds.size.z;
        _CellSize = Mathf.Pow(_BBvolume / (_MaxNeightbors * numberOfBoids), 1f/3f);

        _CellsPerX = (int)(_Bounds.bounds.size.x / _CellSize);
        _CellsPerY = (int)(_Bounds.bounds.size.y / _CellSize);
        _CellsPerZ = (int)(_Bounds.bounds.size.z / _CellSize);

        _TotalCells = _CellsPerX * _CellsPerY * _CellsPerZ;

        //scaling variables with boids sizes to keep behavior the same no matter the size
        float scale = math.max(boidScale.x, boidScale.y) >= math.max(boidScale.x, boidScale.z) ? math.max(boidScale.x, boidScale.y) : boidScale.z;
        fieldOfView *= scale;
        protectedRange *= scale;
        steerAwayFromBoundsStrenght *= scale;

        //i cannot believe that there isn't a simpler round up function
        _NumThreadGroupsForBoidsToDispatch = (int)Math.Ceiling(numberOfBoids/(double)THREADS_PER_GROUP);

        //finding kernel index for the boids logic kernel
        BoidsLogicKernelIndex = _BoidsLogicShader.FindKernel("BoidsLogic");
        SpatialHashingKernelIndex = _BoidsLogicShader.FindKernel("SpatialHashing");

        //initializing I/O buffers
        _InputBuffer = new ComputeBuffer(numberOfBoids, sizeof(float) * 7);
        _OutputBuffer = new ComputeBuffer(numberOfBoids, sizeof(float) * 7);

        _HashTableBuffer = new ComputeBuffer(hashTableSize + 1, sizeof(int));
        _HashStartBuffer = new ComputeBuffer(numberOfBoids, sizeof(int));
        _QueryIDs = new ComputeBuffer(numberOfBoids, sizeof(uint));
        _EmptyQueryBuffer = new ComputeBuffer(numberOfBoids, sizeof(uint));

        //binding I/O buffers
        _BoidsLogicShader.SetBuffer(BoidsLogicKernelIndex, "_InputBoidBuffer", _InputBuffer);
        _BoidsLogicShader.SetBuffer(BoidsLogicKernelIndex, "_OutputBoidBuffer", _OutputBuffer);
        _BoidsLogicShader.SetBuffer(BoidsLogicKernelIndex, "queryIDs", _QueryIDs);
        _BoidsLogicShader.SetBuffer(BoidsLogicKernelIndex, "emptyQueryBuffer", _EmptyQueryBuffer);
        _BoidsLogicShader.SetBuffer(BoidsLogicKernelIndex, "hashTable", _HashTableBuffer);
        _BoidsLogicShader.SetBuffer(BoidsLogicKernelIndex, "hashTableStart", _HashStartBuffer);

        _BoidsLogicShader.SetBuffer(SpatialHashingKernelIndex, "_InputBoidBuffer", _InputBuffer);
        _BoidsLogicShader.SetBuffer(SpatialHashingKernelIndex, "hashTable", _HashTableBuffer);
        _BoidsLogicShader.SetBuffer(SpatialHashingKernelIndex, "hashTableStart", _HashStartBuffer);
        _BoidsLogicShader.SetBuffer(SpatialHashingKernelIndex, "queryIDs", _QueryIDs);

        //binding compute shader vars
        _BoidsLogicShader.SetInt("numOfBoids", numberOfBoids);
        _BoidsLogicShader.SetInt("maxNeightbors", _MaxNeightbors);
        _BoidsLogicShader.SetInt("tableSize", hashTableSize);

        _BoidsLogicShader.SetFloat("fieldOfView", fieldOfView);
        _BoidsLogicShader.SetFloat("protectedRange", protectedRange);
        _BoidsLogicShader.SetFloat("maxSpeed", maxSpeed);

        _BoidsLogicShader.SetFloat("centeringFactor", centeringFactor);
        _BoidsLogicShader.SetFloat("alignmentFactor", alignmentFactor);
        _BoidsLogicShader.SetFloat("avoidanceFactor", avoidanceFactor);
        _BoidsLogicShader.SetFloat("turnFactor", steerAwayFromBoundsStrenght);

        _BoidsLogicShader.SetVector("center", _Bounds.bounds.center);
        _BoidsLogicShader.SetVector("size", _Bounds.bounds.size);
        _BoidsLogicShader.SetVector("maxCorner", _Bounds.bounds.max);
        _BoidsLogicShader.SetVector("minCorner", _Bounds.bounds.min);

        //initialyzing batch renderer
        boidDatas = new BoidData[numberOfBoids];
        Matrix4x4[] matrices = new Matrix4x4[ITEMS_PER_BATCHES];
        batches.Add(matrices);

        //populating boidData array and batch renderer
        int currentBatch = 0;
        int itemCounter = 0;
        for(int currentBoid = 0; currentBoid < numberOfBoids; currentBoid++)
        {
            boidDatas[currentBoid].Position = new float3(Random.Range(_Bounds.bounds.min.x, _Bounds.bounds.max.x), Random.Range(_Bounds.bounds.min.y, _Bounds.bounds.max.y), Random.Range(_Bounds.bounds.min.z, _Bounds.bounds.max.z));
            boidDatas[currentBoid].Velocity = new float3(Random.Range(0, 1), Random.Range(0, 1), Random.Range(0, 1));
            boidDatas[currentBoid].Group = AssignGroup(boidDatas[currentBoid].Position, _Bounds.center);

            batches[currentBatch][itemCounter] = MatrixHelper.MatrixBuilder(boidDatas[currentBoid].Position, quaternion.identity, boidScale);

            itemCounter++;

            if(itemCounter >= ITEMS_PER_BATCHES)
            {
                itemCounter = 0;
                currentBatch++;
                Matrix4x4[] currentmatrices = new Matrix4x4[ITEMS_PER_BATCHES];
                batches.Add(currentmatrices);
            }
        }

        //assigning array to I buffer
        _InputBuffer.SetData(boidDatas);
       }

    void Start()
    {
        Init();
    }

    private void SimStep()
    {
        //dispatching kernels
        _BoidsLogicShader.Dispatch(SpatialHashingKernelIndex,1,1,1);
        _BoidsLogicShader.Dispatch(BoidsLogicKernelIndex, _NumThreadGroupsForBoidsToDispatch, 1, 1);

        //collecting data from shader
        _OutputBuffer.GetData(boidDatas);

        //updating batches for batch renderer
        int id = 0;
        int currentBatch = 0;
        foreach (BoidData boidData in boidDatas)
        {
            batches[currentBatch][id] = Matrix4x4.TRS(boidData.Position, Quaternion.LookRotation(_Camera.transform.forward), boidScale);
            id++;
            if (id >= ITEMS_PER_BATCHES)
            {
                currentBatch++;
                id = 0;
            }
        }

        //assigning collected values from the O buffer to the I buffer
        _InputBuffer.SetData(boidDatas);

        //drawing all boids batches, 1000 at a time per batch (hard limit)
        DrawBoids();
    }
    private void DrawBoids()
    {
        for (int batch = 0; batch < batches.Count; batch++)
        {
            Graphics.DrawMeshInstanced(boidMesh, 0, boidMaterial, batches[batch]);
        }
    }

    void Update()
    {
        SimStep();
    }
}

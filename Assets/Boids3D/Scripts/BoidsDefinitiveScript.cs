using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using Random = UnityEngine.Random;

public struct BoidData
{
    public Vector3 Position;
    public Vector3 Velocity;
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
    public int numOfGroup;


    [Header("sim params")]
    [SerializeField] private ComputeShader _BoidsLogicShader;
    [SerializeField] BoxCollider _Bounds;
    [SerializeField] private int _MaxNeightbors;
    [SerializeField] private Camera _Camera;
    [SerializeField] public int _SimulationSpeed = 1;

    const int THREADS_PER_GROUP = 256;
    const int ITEMS_PER_BATCHES = 1023;
    int _NumThreadGroupsForBoidsToDispatch = 1;
    int BoidsLogicKernelIndex = 0;

    BoidData[] boidDatas;
    Matrix4x4[] boidMatrices;

    List<Matrix4x4[]> batches = new List<Matrix4x4[]>();

    ComputeBuffer _InputBuffer;
    ComputeBuffer _OutputBuffer;

    private int AssignGroup(int pTotalNumOfGroup)
    {
        int group = Random.Range(0, pTotalNumOfGroup);
        return group;
    }

    private void Init()
    {
        //scaling variables with boids sizes to keep behavior the same no matter the size
        float scale = math.max(boidScale.x, boidScale.y) >= math.max(boidScale.x, boidScale.z) ? math.max(boidScale.x, boidScale.y) : boidScale.z;
        fieldOfView *= scale;
        protectedRange *= scale;
        steerAwayFromBoundsStrenght *= scale;

        // Finding Thread Groups
        _NumThreadGroupsForBoidsToDispatch = Mathf.CeilToInt(numberOfBoids/ (float)THREADS_PER_GROUP);

        // finding kernel index for the boids logic kernel
        BoidsLogicKernelIndex = _BoidsLogicShader.FindKernel("BoidsLogic");

        // initializing I/O buffers
        _InputBuffer = new ComputeBuffer(numberOfBoids, sizeof(float) * 7);
        _OutputBuffer = new ComputeBuffer(numberOfBoids, sizeof(float) * 16);

        //binding I/O buffers
        _BoidsLogicShader.SetBuffer(BoidsLogicKernelIndex, "_InputBoidBuffer", _InputBuffer);
        _BoidsLogicShader.SetBuffer(BoidsLogicKernelIndex, "_OutputBuffer", _OutputBuffer);

        //binding compute shader vars
        _BoidsLogicShader.SetInt("numOfBoids", numberOfBoids);
        _BoidsLogicShader.SetInt("maxNeightbors", _MaxNeightbors);

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

        boidMatrices = new Matrix4x4[numberOfBoids];

        //populating boidData array and batch renderer
        int currentBatch = 0;
        int itemCounter = 0;
        for(int currentBoid = 0; currentBoid < numberOfBoids; currentBoid++)
        {
            boidDatas[currentBoid].Position = new Vector3(Random.Range(_Bounds.bounds.min.x, _Bounds.bounds.max.x), Random.Range(_Bounds.bounds.min.y, _Bounds.bounds.max.y), Random.Range(_Bounds.bounds.min.z, _Bounds.bounds.max.z));
            boidDatas[currentBoid].Group = AssignGroup(numOfGroup);

            Matrix4x4 currentMatrix = MatrixHelper.MatrixBuilder(boidDatas[currentBoid].Position, quaternion.identity, boidScale);
            boidMatrices[currentBoid] = currentMatrix;
            batches[currentBatch][itemCounter] = currentMatrix;

            itemCounter++;

            if(itemCounter >= ITEMS_PER_BATCHES)
            {
                int lItemsToBatch = ITEMS_PER_BATCHES;
                itemCounter = 0;
                currentBatch++;

                if (ITEMS_PER_BATCHES >= numberOfBoids - ITEMS_PER_BATCHES * currentBatch) 
                {
                    lItemsToBatch = numberOfBoids - ITEMS_PER_BATCHES * currentBatch;
                }

                Matrix4x4[] currentmatrices = new Matrix4x4[lItemsToBatch];
                batches.Add(currentmatrices);
            }
        }

        //assigning array to I buffer
        _InputBuffer.SetData(boidDatas);
        _OutputBuffer.SetData(boidMatrices);
       }

    void Start()
    {
        Init();
    }

    private void SimStep()
    {
        // Update Dynamic Values
        UpdateSimulationParameters();

        //dispatching kernels
        _BoidsLogicShader.Dispatch(BoidsLogicKernelIndex, _NumThreadGroupsForBoidsToDispatch, 1, 1);

        //collecting data from shader
        _OutputBuffer.GetData(boidMatrices);

        //updating batches for batch renderer
        int id = 0;
        int currentBatch = 0;
        foreach (Matrix4x4 boidData in boidMatrices)
        {
            batches[currentBatch][id] = boidData;
            id++;
            if (id >= ITEMS_PER_BATCHES)
            {
                currentBatch++;
                id = 0;
            }
        }

        //drawing all boids batches, 1000 at a time per batch (hard limit)
        DrawBoids();
    }

    private void UpdateSimulationParameters()
    {
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

    private void OnDrawGizmos()
    {

    }
}

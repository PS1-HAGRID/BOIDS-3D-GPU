using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UIElements;

public class SpatialHashTable
{
    Dictionary<int, List<Vector3>> gridGPU;

    int cellSize;
    int gridWidth;
    int gridHeight;
    int gridLength;

    //debug

    public SpatialHashTable(int cellSize, int width, int height, int lenght) 
    { 
        gridHeight = height;
        gridWidth = width;
        gridLength = lenght;
        this.cellSize = cellSize;
    }

    public void CreateGridGPU()
    {
        int rows = gridHeight / cellSize;
        int columns = gridWidth / cellSize;
        int aisles = gridLength / cellSize;

        int numCells = rows * columns * aisles;

        gridGPU = new Dictionary<int, List<Vector3>>(numCells);

        for (int i = 0; i < numCells; i++)
        {
            gridGPU.Add(GetCellIndex(GetCenterOfCell(i)), new List<Vector3>());
        }

    }

    public void UpdateGridGPU(float3[] objectsToUpdate)
    {
        CreateGridGPU();
        AddAllObjectsToGridGPU(objectsToUpdate);
    }

    public void AddAllObjectsToGridGPU(float3[] objects)
    {
        for (int i = 0; i < objects.Length; i++)
        {
            AddGPU(objects[i]);
        }
    }

    public void ClearGrid() 
    {
        gridGPU.Clear();
    }

    public List<Vector3> QueryGPU(Vector3 position, float FOV)
    {
        List<Vector3> found = new List<Vector3>();
        for (int i = 0; i <= gridGPU.Count; i++)
        {
            if (gridGPU.TryGetValue(i, out var list))
            {
                foreach (var boid in list)
                {
                    if (Vector3.Distance(position, GetCenterOfCell(i)) <= FOV)
                    {
                        found.Add(boid);
                    }
                }
            }
        }

        return found;
    }

    public void AddGPU(Vector3 obj)
    {
        int cellIndex = GetCellIndex(obj);
        if (!gridGPU.TryGetValue(cellIndex, out var list))
        {
            list = new List<Vector3>();
            gridGPU[cellIndex] = list;
        }
        list.Add(obj);
    }

    private Vector3 GetCenterOfCell(int cellIndex)
    {
        float x = cellIndex % cellSize;
        float y = (cellIndex/cellSize) % cellSize;
        float z = (cellIndex/ (cellSize * cellSize));
        Vector3 center = new Vector3(x,y,z) * cellSize;
        return center;
    }

    public int GetCellIndex(Vector3 target)
    {
        return (int)(((target.z * cellSize * cellSize) + (target.y * cellSize) + target.x)/cellSize);
    }



}

using Unity.Mathematics;
public class SpatialHash {

    public SpatialHash(float pCellSize, int pNumOfElement, float3[] pPositions)
    {
        _CellSize = pCellSize;
        _NumOfElement = pNumOfElement;
        _Positions = pPositions;

        InitTable();
    }

    private float3[] _Positions;

    public int[] queryIDs;
    private int[] _HashTable;
    private int[] _HashStart;

    private float _CellSize;
    private int _NumOfElement;
    private int _TableSize;
    private int _NumOfQueriedObject;

    //hashTable initialization method
    private void InitTable()
    {
        //this can change depending on the context
        _TableSize = _NumOfElement * 2;

        _HashStart = new int[_TableSize + 1]; //the _TableSize + 1 is to make a guard, it's kinda redundant
        _HashTable = new int[_NumOfElement];
        queryIDs = new int[_NumOfElement];

        CreateTable();
    }

    //magic hash function that gives an almost unique ID (hash) within the bounds of the hash table 
    private int HashFunction(int3 pGridPos)
    {
        return math.abs((pGridPos.x * 92837111 * 3) ^ (pGridPos.y * 689287499 * 5) ^ (pGridPos.z * 283923481 * 7)) % _TableSize;
    }

    //simple position to grid coordinate function
    private int3 GetGridCoords(float3 pPosition)
    {
        return new int3(math.floor(pPosition / _CellSize));
    }

    // the query getter
    public int[] GetQueryIDs(float3 objectToQuery, float range)
    {
        return Query(objectToQuery,range);
    }

    //this one just makes my life easier and looks clean in the code tbh
    private int GetHash(float3 pPosition)
    {
        return HashFunction(GetGridCoords(pPosition));
    }

    //main hash table population method
    private void CreateTable()
    {
        //finds bucket sizes
        for (int objectID = 0; objectID < _NumOfElement; objectID++) 
        {
            int lCurrentHash = GetHash(_Positions[objectID]);
            _HashStart[lCurrentHash]++;
        }
        
        //finds the start of the table
        int start = 0;
        for (int HashID = 0; HashID < _TableSize; HashID++) 
        {
            start += _HashStart[HashID];
            _HashStart[HashID] = start;
        }
        _HashStart[_TableSize] = start; // guard
        
        //fills in table
        for(int objectID = 0; objectID < _NumOfElement; objectID++)
        {
            int lCurrentHash = GetHash(_Positions[objectID]);
            _HashStart[lCurrentHash]--;
            _HashTable[_HashStart[lCurrentHash]] = objectID;
        }
    }

    //main Query function
    private int[] Query(float3 pObjectToQueryPos, float pRange)
    {
        //find the start and the end of the query
        int3 lQueryStart = GetGridCoords(pObjectToQueryPos - pRange);
        int3 lQueryEnd = GetGridCoords(pObjectToQueryPos + pRange);
        
        _NumOfQueriedObject = 0;

        //ooooh baby that's a big ugly nest
        for (int XQuery = lQueryStart.x; XQuery < lQueryEnd.x; XQuery++) 
        { 
            for(int YQuery = lQueryStart.y; YQuery < lQueryEnd.y; YQuery++)
            {
                for( int ZQuery = lQueryStart.z; ZQuery < lQueryEnd.z; ZQuery++)
                {
                    int CurrentGridHash = GetHash(new float3(XQuery, YQuery, ZQuery));
                    int start = _HashStart[CurrentGridHash];
                    int end = _HashStart[CurrentGridHash + 1];

                    for(int range = start; range < end; range++)
                    {
                        //the problem child, Y U NO WORK (return index out of range)
                        queryIDs[_NumOfQueriedObject] = _HashTable[range];
                        _NumOfQueriedObject++;
                    }
                }
            }
        }
        
        return queryIDs;
    }

}

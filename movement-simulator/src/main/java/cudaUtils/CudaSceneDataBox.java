package cudaUtils;

import javafx.geometry.Pos;
import utility.Position;

import java.util.ArrayList;
import java.util.List;

public class CudaSceneDataBox {
    private CudaSceneMetadata cudaSceneMetadata;
    private List<List<Position>> paths;

    public void setCudaSceneMetadata(CudaSceneMetadata cudaSceneMetadata) {
        this.cudaSceneMetadata = cudaSceneMetadata;
    }

    public CudaSceneMetadata getCudaSceneMetadata() {
        return cudaSceneMetadata;
    }

    public List<Position> getStartPositions() {
        List<Position> startPositions = new ArrayList<>();
        for (List<Position> agentPath : paths)
            startPositions.add(agentPath.get(0));
        return startPositions;
    }

    public void setPaths(List<List<Position>> paths) {
        this.paths = paths;
    }

    public List<List<Position>> getPaths() {
        return paths;
    }
}

package cudaUtils;

import utility.Position;

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
        return paths.get(0);
    }

    public void setPaths(List<List<Position>> paths) {
        this.paths = paths;
    }

    public List<List<Position>> getPaths() {
        return paths;
    }
}

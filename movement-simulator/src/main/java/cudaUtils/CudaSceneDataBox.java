package cudaUtils;

import javafx.geometry.Pos;
import utility.Position;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CudaSceneDataBox {
    private CudaSceneMetadata cudaSceneMetadata;
    private CudaSceneDataImporter importer;

    public CudaSceneDataBox() throws IOException {
        this.importer = new CudaSceneDataImporter();
        this.cudaSceneMetadata = importer.getCudaSceneMetadata();
    }

    public CudaSceneMetadata getCudaSceneMetadata() throws IOException {
        return cudaSceneMetadata;
    }

    public List<Position> getNextPositionsList() throws IOException {
        return importer.getNextPositionsList();
    }
}

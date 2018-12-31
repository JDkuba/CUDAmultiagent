package cudaUtils;

import javafx.geometry.Pos;
import utility.Position;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class CudaSceneDataImporter {

    static public final String METADATA_FILE_NAME = "data/metadata.out";
    static public final String PATHS_FILE_PATH = "data/agents_positions.out";

    static public CudaSceneMetadata getCudaSceneMetadata() throws IOException {
        Path filePath = Paths.get(".").toAbsolutePath().getParent().getParent().resolve(METADATA_FILE_NAME);
        Scanner scanner = new Scanner(filePath);
        return new CudaSceneMetadata()
                        .setAgentNumber(scanner.nextInt())
                        .setAgentRadius(scanner.nextDouble())
                        .setBoardX(scanner.nextInt())
                        .setBoardY(scanner.nextInt());
    }

    static public List<List<Position>> getAgentsPaths() throws IOException {

        Path filePath = Paths.get(".").toAbsolutePath().getParent().getParent().resolve(PATHS_FILE_PATH);
        Scanner scanner = new Scanner(filePath);
        int agentsNumber = scanner.nextInt();
        List<List<Position>> paths = new ArrayList<>();
        for (int i = 0; i < agentsNumber; i++)
            paths.add(new ArrayList<>());
        while (scanner.hasNext()) {
            for(List<Position> agentPath : paths)
                agentPath.add(new Position(scanner.nextDouble(), scanner.nextDouble()));
        }
        scanner.close();
        return paths;
    }

    static public CudaSceneDataBox getCudaSceneData() throws IOException {
        CudaSceneDataBox sceneDataBox = new CudaSceneDataBox();
        sceneDataBox.setPaths(getAgentsPaths());
        sceneDataBox.setCudaSceneMetadata(getCudaSceneMetadata());
        return sceneDataBox;
    }
}

package cudaUtils;

import javafx.geometry.Pos;
import utility.Position;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

class CudaSceneDataImporter {

    static public final String METADATA_FILE_NAME = "data/metadata.out";
    static public final String AGENTS_POSTITION_FILE_PATH = "data/agents_positions.out";
    static public final int DSIZE = 4;

    private Integer agentsNumber;
    private InputStream positionsStream;
    private File positionsFile;

    public CudaSceneMetadata getCudaSceneMetadata() throws IOException {
        Path filePath = Paths.get(".").toAbsolutePath().getParent().getParent().resolve(METADATA_FILE_NAME);
        Scanner scanner = new Scanner(filePath);
        return new CudaSceneMetadata()
                        .setAgentNumber(scanner.nextInt())
                        .setGenerationsNumber(scanner.nextInt())
                        .setAgentRadius(scanner.nextDouble())
                        .setBoardX(scanner.nextInt())
                        .setBoardY(scanner.nextInt());
    }

    private void openPositionsResources() throws FileNotFoundException {
        positionsFile = Paths.get(".").toAbsolutePath().getParent().getParent().resolve(AGENTS_POSTITION_FILE_PATH).toFile();
        positionsStream = new FileInputStream(positionsFile);
    }

    private void closePositionResources() throws IOException {
        positionsStream.close();
    }

    public List<Position> getNextPositionsList() throws IOException{
        if(agentsNumber == null){
            openPositionsResources();
            byte[] bytes = new byte[DSIZE];
            positionsStream.read(bytes);
            agentsNumber = (int) bytes[0];
        }
        byte[] bytes = new byte[2*DSIZE*agentsNumber];
        if(positionsStream.read(bytes) == -1){
            closePositionResources();
            return null;
        }
        List<Position> positions = new ArrayList<>();
        for (int i = 0; i < 2*DSIZE*agentsNumber; i+=2*DSIZE)
            positions.add(new Position((int) bytes[i], (int) bytes[i+DSIZE]));
        return positions;
    }
}

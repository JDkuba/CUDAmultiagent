package cudaUtils;

import javafx.geometry.Pos;
import utility.Position;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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
            agentsNumber = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getInt();
            System.out.println(agentsNumber);
        }
        byte[] bytes = new byte[2*DSIZE*agentsNumber];
        if(positionsStream.read(bytes) == -1){
            closePositionResources();
            return null;
        }
        List<Position> positions = new ArrayList<>();
        for (int i = 0; i < 2*DSIZE*agentsNumber; i+=2*DSIZE){
            int x = ByteBuffer.wrap(bytes, i, DSIZE).order(ByteOrder.LITTLE_ENDIAN).getInt();
            int y = ByteBuffer.wrap(bytes, i+DSIZE, DSIZE).order(ByteOrder.LITTLE_ENDIAN).getInt();
            positions.add(new Position(x, y));
        }
        return positions;
    }
}

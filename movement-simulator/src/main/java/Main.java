import controller.Controller;
import cudaUtils.CudaSceneDataBox;
import cudaUtils.CudaSceneMetadata;
import javafx.application.Application;
import javafx.stage.Stage;
import utility.Position;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main extends Application {

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage){
        Controller controller = new Controller(primaryStage);
        try {
            CudaSceneDataBox dataBox = new CudaSceneDataBox();
            int gens = dataBox.getCudaSceneMetadata().getGenerationsNumber();
            List<List<Position>> list = new ArrayList<>();
            for (int i = 0; i < dataBox.getCudaSceneMetadata().getGenerationsNumber(); i++)
                list.add(dataBox.getNextPositionsList());
            System.out.println(list);
            System.out.println(list.size());
            System.out.println(gens);
//            controller.setAnimation(getCudaSceneData());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

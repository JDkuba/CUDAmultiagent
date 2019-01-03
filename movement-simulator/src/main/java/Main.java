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
            controller.setAnimation(new CudaSceneDataBox());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

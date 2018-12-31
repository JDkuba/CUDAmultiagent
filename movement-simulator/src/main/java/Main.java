import controller.Controller;
import javafx.application.Application;
import javafx.stage.Stage;

import java.io.IOException;

import static cudaUtils.CudaSceneDataImporter.getCudaSceneData;

public class Main extends Application {

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage){
        Controller controller = new Controller(primaryStage);
        try {
            controller.setAnimation(getCudaSceneData());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

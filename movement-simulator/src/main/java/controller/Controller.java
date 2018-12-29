package controller;

import javafx.stage.Stage;
import view.View;

public class Controller {
    private interfaces.View view;
    public Controller(Stage primaryStage) {
        this.view = new View(primaryStage);
    }

    public void run() {

    }
}

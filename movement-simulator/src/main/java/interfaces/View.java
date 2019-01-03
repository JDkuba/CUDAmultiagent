package interfaces;

import abstractClasses.AbstractPlane;
import javafx.stage.Stage;
import utility.Position;

import java.util.List;

public interface View {
    AbstractPlane getPlane();
    void setPlaneSize(double width, double height);
}

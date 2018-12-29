package abstractClasses;

import javafx.scene.layout.Pane;
import utility.Position;

import java.util.List;

public abstract class AbstractPlane extends Pane {
    public abstract void addAgents(List<Position> positions);
    public abstract void moveAgents(List<Position> positions);
}

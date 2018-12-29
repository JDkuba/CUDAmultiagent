package interfaces;

import javafx.stage.Stage;
import utility.Position;

import java.util.List;

public interface View {
    void addAgents(List<Position> positions);
    void moveAgents(List<Position> positions);
}

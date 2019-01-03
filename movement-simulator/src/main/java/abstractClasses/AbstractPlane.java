package abstractClasses;

import javafx.animation.Transition;
import javafx.scene.layout.Pane;
import utility.Position;

import java.util.List;

public abstract class AbstractPlane extends Pane {
    public abstract void addAgents(List<Position> positions);
    public abstract void setAgentsPositions(List<Position> positions);
    public abstract void setAgentsSize(double size);
    public abstract void addAgentsPathTranslation(List<List<Position>> agentMovements);
    public abstract void stopAgentSimulation();
    public abstract void playAgentSimulation();
    public abstract void playAgentSimulationFromStart();
}

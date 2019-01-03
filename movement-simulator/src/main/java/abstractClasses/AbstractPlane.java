package abstractClasses;

import javafx.animation.Transition;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.util.Duration;
import utility.Position;

import java.util.List;

public abstract class AbstractPlane extends Pane {
    public abstract void addAgents(List<Position> positions);
    public abstract void setAgentsPositions(List<Position> positions);
    public abstract void addAgentsPathTranslation(List<List<Position>> agentMovements);
    public abstract void stopAgentSimulation();
    public abstract void playAgentSimulation();
    public abstract void playAgentSimulationFromStart();
    public abstract void setAgentsRadius(double radius);
    public abstract void setAgentsRadii(List<Double> radii);
    public abstract void setAgentsColor(Color color);
    public abstract void setAgentsColors(List<Color> colors);
    public abstract void setBackgroundSize(double width, double height);
    public abstract void setBackgroundColor(Color color);
    public abstract void setFrameDuration(Duration duration);
}

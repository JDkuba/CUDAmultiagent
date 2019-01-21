package view;

import abstractClasses.AbstractPlane;
import javafx.animation.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.util.Duration;
import utility.Position;
import view.drawable.Agent;
import view.drawable.MovablePane;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;

class Plane extends AbstractPlane {
    private List<Agent> agents;
    private Rectangle background;
    private List<List<List<Position>>> transitionList;
    private Transition[] transitions;
    private int currentTransitionNumber;

    private double agentRadius;
    private Duration frameDuration;
    private Color agentColor;

    Plane(double width, double height) {
        super();
        initializeBackground(width, height);
        this.agents = new ArrayList<>();
        this.agentRadius = Config.STANDARD_AGENT_RADIUS;
        this.frameDuration = Config.STANDARD_FRAME_DURATION;
        this.agentColor = Config.STANDARD_AGENT_COLOR;
        this.transitionList = new ArrayList<>();
        this.transitions = new Transition[3];
    }

    private void initializeBackground(double width, double height) {
        background = new Rectangle(width, height);
        background.setFill(Config.STANDARD_PLANE_BACKGROUND_COLOR);
        this.getChildren().add(background);
    }

    @Override
    public void addAgents(List<Position> positions) {
        for (Position position : positions) {
            Agent agent = new Agent(position, agentRadius);
            agent.setColor(agentColor);
            agents.add(agent);
            this.getChildren().add(agent);
        }
    }

    @Override
    public void setAgentsPositions(List<Position> positions) {
        for (int i = 0; i < positions.size(); i++) {
            agents.get(i).setTranslateX(positions.get(i).getX());
            agents.get(i).setTranslateY(positions.get(i).getY());
        }
    }

    @Override
    public void setAgentsRadius(double radius) {
        agentRadius = radius;
        for (Agent agent : agents) {
            agent.setRadius(radius);
        }
    }

    @Override
    public void setAgentsRadii(List<Double> radii) {
        IntStream.range(0, radii.size()).forEach(i -> agents.get(i).setRadius(radii.get(i)));
    }

    @Override
    public void setAgentsColor(Color color) {
        for (Agent agent : agents) {
            agent.setColor(color);
        }
    }

    @Override
    public void setAgentsColors(List<Color> colors) {
        IntStream.range(0, colors.size()).forEach(i -> agents.get(i).setColor(colors.get(i)));
    }

    @Override
    public void setBackgroundSize(double width, double height) {
        background.setWidth(width);
        background.setHeight(height);
    }

    @Override
    public void setBackgroundColor(Color color) {
        background.setFill(color);
    }

    @Override
    public void setFrameDuration(Duration duration) {
        this.frameDuration = duration;
    }

    @Override
    public void addAgentsPathTranslation(List<List<Position>> agentsPositions) {
        transitionList.add(agentsPositions);
        if (transitionList.size() <= transitions.length) {
            initializeCurrentTransition();
        }
    }

    @Override
    public void stopAgentSimulation() {
        transitions[0].stop();
    }

    @Override
    public void playAgentSimulation() {
        transitions[0].play();
    }

    @Override
    public void playAgentSimulationFromStart() {
        transitions[0].stop();
        initializeCurrentTransition();
        transitions[0].playFromStart();
    }

    @Override
    public void changeAgentSimulationPlayStatus() {
        if (transitions[0].getStatus().compareTo(Animation.Status.RUNNING) == 0) {
            transitions[0].stop();
        } else {
            transitions[0].play();
        }
    }

    private void initializeCurrentTransition() {
        currentTransitionNumber = 1;
        for (int i = 0; i < transitions.length; i++) {
            if (i < transitionList.size()) {
                transitions[i] = getPathTranslateAgents(transitionList.get(i));
                setEventHandler(transitions[i]);
            }
        }
    }

    private void setEventHandler(Transition transition) {
        transition.setOnFinished(event -> {
            if (currentTransitionNumber < transitionList.size()) {
                transitions[1].play();

                System.arraycopy(transitions, 1, transitions, 0, transitions.length - 1);

                currentTransitionNumber++;
                if (currentTransitionNumber + transitions.length - 2 < transitionList.size()) {
                    transitions[transitions.length - 1] = getPathTranslateAgents(transitionList.get(currentTransitionNumber + transitions.length - 2));
                    setEventHandler(transitions[transitions.length - 1]);
                }
            }
        });
    }

    private Transition getPathTranslateAgents(List<List<Position>> agentsPositions) {

        ParallelTransition parallelTransition = new ParallelTransition();

        List<SequentialTransition> sequentialTransitions = Collections.synchronizedList(new ArrayList<>());

        IntStream.range(0, agents.size()).parallel().forEach((agentNumber) -> {
            SequentialTransition sequentialTransition = new SequentialTransition();
            Agent agent = agents.get(agentNumber);
            ArrayList<TranslateTransition> transitions = new ArrayList<>();

            IntStream.range(0, agentsPositions.size()).forEach( i -> {
                TranslateTransition translateTransition = new TranslateTransition(
                        this.frameDuration, agent);
                translateTransition.setToX(agentsPositions.get(i).get(agentNumber).getX());
                translateTransition.setToY(agentsPositions.get(i).get(agentNumber).getY());
                transitions.add(translateTransition);
            });
            sequentialTransition.getChildren().addAll(transitions);
            sequentialTransitions.add(sequentialTransition);
        });

        parallelTransition.getChildren().addAll(sequentialTransitions);
        return parallelTransition;
    }
}

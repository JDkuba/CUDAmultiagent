package view;

import abstractClasses.AbstractPlane;
import javafx.animation.*;
import javafx.scene.shape.Rectangle;
import utility.Position;
import view.drawable.Agent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;

class Plane extends AbstractPlane {
    private List<Agent> agents;
    private Rectangle background;
    private double agentSize;
    private List<List<List<Position>>> transitionList;
    private Transition[] transitions;
    private int currentTransitionNumber;
    Plane(double width, double height) {
        super();
        initializeBackground(width, height);
        this.agents = new ArrayList<>();
        this.agentSize = Config.AGENT_SIZE;
        transitionList = new ArrayList<>();
        transitions = new Transition[3];
    }

    private void initializeBackground(double width, double height) {
        background = new Rectangle(width, height);
        background.setFill(Config.PLANE_BACKGROUND_COLOR);
        this.getChildren().add(background);
    }

    @Override
    public void addAgents(List<Position> positions) {
        for (Position position : positions) {
            Agent agent = new Agent(position, agentSize);
            agent.setColor(Config.STANDARD_AGENT_COLOR);
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
    public void setAgentsSize(double size) {
        agentSize = size;
        for (Agent agent : agents) {
            agent.setSize(size);
        }
    }

    @Override
    public void addAgentsPathTranslation(List<List<Position>> agentMovements) {
        transitionList.add(agentMovements);
        if (transitionList.size() == transitions.length) {
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

    private void initializeCurrentTransition() {
        currentTransitionNumber = 1;
        for (int i = 0; i < transitions.length; i++) {
            transitions[i] = getPathTranslateAgents(transitionList.get(i));
            setEventHandler(transitions[i]);
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

    private Transition getPathTranslateAgents(List<List<Position>> agentMovements) {

        ParallelTransition parallelTransition = new ParallelTransition();

        List<SequentialTransition> sequentialTransitions = Collections.synchronizedList(new ArrayList<>());

        IntStream.range(0, agentMovements.size()).parallel().forEach((index) -> {
            SequentialTransition sequentialTransition = new SequentialTransition();
            Agent agent = agents.get(index);
            ArrayList<TranslateTransition> transitions = new ArrayList<>();

            for (Position position : agentMovements.get(index)) {
                TranslateTransition translateTransition = new TranslateTransition(
                        Config.AGENT_TRANSITION_FRAME_DURATION, agent);
                translateTransition.setToX(position.getX());
                translateTransition.setToY(position.getY());
                transitions.add(translateTransition);
            }
            sequentialTransition.getChildren().addAll(transitions);
            sequentialTransitions.add(sequentialTransition);
        });

        parallelTransition.getChildren().addAll(sequentialTransitions);
        return parallelTransition;
    }
}

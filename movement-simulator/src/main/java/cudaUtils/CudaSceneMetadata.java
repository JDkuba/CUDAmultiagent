package cudaUtils;

public class CudaSceneMetadata {
    private int agentNumber;
    private int generationsNumber;
    private double agentRadius;
    private int boardX;
    private int boardY;

    public CudaSceneMetadata setAgentNumber(int agentNumber) {
        this.agentNumber = agentNumber;
        return this;
    }

    public int getAgentNumber() {
        return agentNumber;
    }

    public CudaSceneMetadata setAgentRadius(double agentRadius) {
        this.agentRadius = agentRadius;
        return this;
    }

    public double getAgentRadius() {
        return agentRadius;
    }

    public CudaSceneMetadata setBoardX(int boardX) {
        this.boardX = boardX;
        return this;
    }

    public int getBoardX() {
        return boardX;
    }

    public CudaSceneMetadata setBoardY(int boardY) {
        this.boardY = boardY;
        return this;
    }

    public int getBoardY() {
        return boardY;
    }

    public CudaSceneMetadata setGenerationsNumber(int generationsNumber) {
        this.generationsNumber = generationsNumber;
        return this;
    }

    public int getGenerationsNumber() {
        return generationsNumber;
    }
}

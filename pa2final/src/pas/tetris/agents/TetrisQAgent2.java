package src.pas.tetris.agents;


import java.util.Collections;
import java.util.HashMap;
// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;


// JAVA PROJECT IMPORTS
import edu.bu.tetris.agents.QAgent;
import edu.bu.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.tetris.game.Board;
import edu.bu.tetris.game.Game.GameView;
import edu.bu.tetris.game.minos.Mino;
import edu.bu.tetris.linalg.Matrix;
import edu.bu.tetris.nn.Model;
import edu.bu.tetris.nn.LossFunction;
import edu.bu.tetris.nn.Optimizer;
import edu.bu.tetris.nn.models.Sequential;
import edu.bu.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.tetris.nn.layers.ReLU;  // some activations (below too)
import edu.bu.tetris.nn.layers.Tanh;
import edu.bu.tetris.nn.layers.Sigmoid;
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Pair;

public class TetrisQAgent
    extends QAgent
{
    private int NumMoves = 0;
    public static final double INITIAL_EXPLORATION_PROB = 1.0;
    public static final double FINAL_EXPLORATION_PROB = 0.005;
    public static final double EXPLORATION_DECAY_RATE = 0.000001;
    public static boolean Lostgame = false;
    private Random random;

    public TetrisQAgent(String name)
    {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() { return this.random; }

    private Map<String, Map<String, Integer>> visitCountMap = new HashMap<>();


    @Override
    public Model initQFunction()
    {
        // build a single-hidden-layer feedforward network
        // this example will create a 3-layer neural network (1 hidden layer)
        // in this example, the input to the neural network is the
        // image of the board unrolled into a giant vector

        final int numPixelsInImage = Board.NUM_ROWS * Board.NUM_COLS;
        final int hiddenDim = 2 * numPixelsInImage;
        final int outDim = 1;

        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(numPixelsInImage, hiddenDim));
        qFunction.add(new Tanh());
        qFunction.add(new Dense(hiddenDim, outDim));
        

        return qFunction;
    }

    // Helper function for numericalize mino orientation
    // A - 0, B - 1, C - 2, D - 3
    private double num_ori(Mino.Orientation ori) {
        double ori_d = -1.0;

        if (ori == Mino.Orientation.A) {
            ori_d = 0.0;
        } else if (ori == Mino.Orientation.B) {
            ori_d = 1.0;
        } else if (ori == Mino.Orientation.C) {
            ori_d = 2.0;
        } else if (ori == Mino.Orientation.D) {
            ori_d = 3.0;
        }
        return ori_d;
    }

    // Helper function for numericalize mino type
    // I - 0, J - 1, L - 2, O - 3, S - 4, T - 5, Z - 6
    private double num_type(Mino.MinoType type) {
        double type_d = -1.0;

        if (type == Mino.MinoType.I) {
            type_d = 0.0;
        } else if (type == Mino.MinoType.J) {
            type_d = 1.0;
        } else if (type == Mino.MinoType.L) {
            type_d = 2.0;
        } else if (type == Mino.MinoType.O) {
            type_d = 3.0;
        } else if (type == Mino.MinoType.S) {
            type_d = 4.0;
        } else if (type == Mino.MinoType.T) {
            type_d = 5.0;
        } else if (type == Mino.MinoType.Z) {
            type_d = 6.0;
        }
        return type_d;
    }

    /**
        This function is for you to figure out what your features
        are. This should end up being a single row-vector, and the
        dimensions should be what your qfunction is expecting.
        One thing we can do is get the grayscale image
        where squares in the image are 0.0 if unoccupied, 0.5 if
        there is a "background" square (i.e. that square is occupied
        but it is not the current piece being placed), and 1.0 for
        any squares that the current piece is being considered for.
        
        We can then flatten this image to get a row-vector, but we
        can do more than this! Try to be creative: how can you measure the
        "state" of the game without relying on the pixels? If you were given
        a tetris game midway through play, what properties would you look for?
     */
    @Override
    public Matrix getQFunctionInput(final GameView game,
                                    final Mino potentialAction)
    {
        // Matrix flattenedImage = null;
        // try
        // {
        //     flattenedImage = game.getGrayscaleImage(potentialAction).flatten();
        // } catch(Exception e)
        // {
        //     e.printStackTrace();
        //     System.exit(-1);
        // }
        // return flattenedImage;

        if(game.didAgentLose() == true){
            System.out.println("Game Over");
            Lostgame = true;
        }
        double[] features = new double[8];
        int i = 0;

        // Orientation of mino
        double ori_d = num_ori(potentialAction.getOrientation());
        features[i] = ori_d;
        i++;

        // Type of mino
        double type_d = num_type(potentialAction.getType());
        features[i] = type_d;
        i++;
        
        // x, y coordinate of mino
        double piv_x = (double)potentialAction.getPivotBlockCoordinate().getXCoordinate();
        double piv_y = (double)potentialAction.getPivotBlockCoordinate().getYCoordinate();
        features[i] = piv_x;
        i++;
        features[i] = piv_y;
        i++;
        
        // Next three mino types
        List<Mino.MinoType> next_three = game.getNextThreeMinoTypes();
        for (Mino.MinoType type : next_three) {
            features[i] = num_type(type);
            i++;
        }

        // Score for this turn
        double score_this_turn = (double)game.getScoreThisTurn();
        features[i] = score_this_turn;
        i++;

        // Matrix for my features
        Matrix m = Matrix.zeros(1, 8);
        for (int col = 0; col < features.length; col++) {
            m.set(0, col, features[col]);
        }

        // Matrix for gray scale images
        Matrix flattenedImage = null;
        Matrix final_m = null;
        try {
            flattenedImage = game.getGrayscaleImage(potentialAction).flatten();
            Shape final_shape = flattenedImage.getShape();
            // Combine both matrix for final feature matrix
            final_m = Matrix.zeros(1, final_shape.getNumCols() + 8);
            final_m.copySlice(0, 1, final_shape.getNumCols(), final_shape.getNumCols() + 8, m);

            // System.out.println(final_m);
        } catch(Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        // return final_m;
        return final_m;
    }
    private String gameToString(final GameView game, final Mino potentialAction) {
        Matrix state = null;
        try {
            state = game.getGrayscaleImage(potentialAction);  // This method might throw an exception
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        StringBuilder g = new StringBuilder();
        int rows = Board.NUM_ROWS;
        int cols = Board.NUM_COLS;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                g.append(state.get(i, j) > 0 ? "1" : "0");
            }
        }
        return g.toString();
    }

    public double getQValue(GameView game, Mino action) {
        // Preprocess the game state and the action into an input vector for the neural network
        Matrix inputVector = getQFunctionInput(game, action);
        Matrix qValue = null;        // Pass the input vector through the network to get the Q-value
        try {
            qValue = this.getQFunction().forward(inputVector); // This method might throw an exception
        } catch (Exception e) {
            System.err.println("Failed to forward input vector: " + e.getMessage());
            // Handle the error or rethrow as unchecked
            throw new RuntimeException("Error during network forward pass", e);
        }
        // Return the Q-value
        return qValue.get(0, 0);
    }
    /**
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good" actions
     * over and over again. This can prevent us from discovering new, potentially even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of the time.
     * While this strategy is easy to implement, it often doesn't perform well and is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */
    @Override
    public boolean shouldExplore(final GameView game,
                                 final GameCounter gameCounter)
    {
        NumMoves++;
        if (INITIAL_EXPLORATION_PROB - EXPLORATION_DECAY_RATE * (NumMoves - 1) < FINAL_EXPLORATION_PROB) {
            return this.getRandom().nextDouble() <= FINAL_EXPLORATION_PROB;
        }

        else {
            return this.getRandom().nextDouble() <= INITIAL_EXPLORATION_PROB - EXPLORATION_DECAY_RATE * (NumMoves - 1);
        }
    }
    // public Mino getExplorationMove(final GameView game) {
    //     // Generate a list of possible actions (e.g., all possible rotations and placements of the current Tetrimino)
    //     List<Mino> possibleActions = generatePossibleActions(game);
    
    //     // Choose a random action from the list of possible actions for exploration
    //     return possibleActions.get(getRandom().nextInt(possibleActions.size()));
    // }
    @Override
    public Mino getExplorationMove(final GameView game)
    {
        List<Mino> possibleMoves = game.getFinalMinoPositions();
        if (possibleMoves.isEmpty()) {
            return null; // No moves available, return null
        }
    
        Mino selectedMove = null;
        double highestExplorationValue = Double.NEGATIVE_INFINITY;
    
        for (Mino move : possibleMoves) {
            String stateKey = gameToString(game,move);
            String actionKey = actionToString(move);
    
            double qValue = getQValue(game, move);
            int visitCount = VisitCountM(stateKey, actionKey);
    
            double explorationValue = qValue + (1.0 / Math.max(1, visitCount));
    
            if (explorationValue > highestExplorationValue) {
                highestExplorationValue = explorationValue;
                selectedMove = move;
            }
        }
    
        if (selectedMove != null) {
            String stateKey = gameToString(game,selectedMove);
            String actionKey = actionToString(selectedMove);
            updateVisitCount(stateKey, actionKey);
        }
    
        return selectedMove;
    }
    private void updateVisitCount(String stateKey, String actionKey) {
        visitCountMap.putIfAbsent(stateKey, new HashMap<>());
        Map<String, Integer> actionMap = visitCountMap.get(stateKey);
        actionMap.put(actionKey, actionMap.getOrDefault(actionKey, 0) + 1);
    }

    private int VisitCountM(String stateKey, String actionKey) {
        return visitCountMap.getOrDefault(stateKey, new HashMap<>())
                            .getOrDefault(actionKey, 0);
    }

    /**
     * This method is called by the TrainerAgent after we have played enough training games.
     * In between the training section and the evaluation section of a phase, we need to use
     * the exprience we've collected (from the training games) to improve the q-function.
     *
     * You don't really need to change this method unless you want to. All that happens
     * is that we will use the experiences currently stored in the replay buffer to update
     * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
     * (i.e. all at once)...this often works better and is an active area of research.
     *
     * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
     * of epochs in between the training and eval sections of each phase.
     */
    @Override
    public void trainQFunction(Dataset dataset,
                               LossFunction lossFunction,
                               Optimizer optimizer,
                               long numUpdates)
    {
        for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx)
        {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix> > batchIterator = dataset.iterator();

            while(batchIterator.hasNext())
            {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try
                {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                                                  lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch(Exception e)
                {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }
    /**
     * This method is where you will devise your own reward signal. Remember, the larger
     * the number, the more "pleasurable" it is to the model, and the smaller the number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced by the
     * points, however this is not all. In fact, just using the points earned this turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally the worse
     * (unless you have a long hole waiting for an I-block). When you design a reward
     * signal that is less sparse, you should see your model optimize this reward over time.
     */
    @Override
    public double getReward(final GameView game) {
        double reward = 0.00;
        Board board = game.getBoard();

        reward += NumMoves*10; // Reward for each move made without losing
        // Reward based on the number of lines cleared in this turn
        int linesCleared = game.getLinesClearedThisTurn();
        reward += linesCleared * 1000000; //If we clear lines, we get a big reward

        // Penalty for the height of the highest column
        int maxHeight = game.getMaxColumnHeight();
        reward -= Math.pow(maxHeight, 2); // Higher stacks are penalized

        // Penalty for number of holes in the board
        int numHoles = game.getNumHoles();
        reward -= Math.pow(numHoles * 10, 2); // Each hole incurs a penalty

        // Bonus for creating complete rows (encourages clearing lines)
        int completeRows = game.getNumCompleteRows();
        reward += completeRows * 200000; // Bonus for each complete row
        for(int i = 0; i < 10; i++) {
            // Penalty for losing (if the top row is occupied, we lose
        if ((board.isCoordinateOccupied(i, 0))) {
            reward -= 1000000; // Big penalty for losing
        }
    }
        return reward;
    }
}

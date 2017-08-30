using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK.CSTrainingExamples
{
    public class Sequential
    {
        public Sequential(IDictionary<string, object> defaultOptions)
        {
            this.defaultOptions = defaultOptions;
        }

        public Sequential Dense(NDShape shape, Activation activation = Activation.None, CNTKDictionary initializer = null, 
            int input_rank = 0, int map_rank = 0, bool bias = true, float init_bias = 0, string name = "")
        {
            throw new NotImplementedException();
        }
        public Sequential Convolution(NDShape kernerShape, int outputChannels, NDShape stride = null, Activation activation = Activation.None)
        {
            throw new NotImplementedException();
        }

        public Sequential Dropout(double dropoutProb = 0.5)
        {
            throw new NotImplementedException();
        }

        public static implicit operator Function(Sequential sequential)
        {
            return sequential.Model;
        }

        public static implicit operator Variable(Sequential sequential)
        {
            return sequential.Model;
        }

        public IList<Parameter> Parameters
        {
            get { return this.Model.Parameters(); }
        }

        public Function Model
        {
            get;
            private set;
        }

        private IDictionary<string, object> defaultOptions;
        private readonly int[] defaultStride = new int[]{ 2, 2 };
    }

    public class ExampleUseCases
    {
        public void Example1()
        {
            // 1. prepare input variables 
            var imageDim = new int[] { 32, 32 };
            var outputDim = new int[] { 10 };
            Variable images = Variable.InputVariable(imageDim, DataType.Float);
            Variable labels = Variable.InputVariable(outputDim, DataType.Float);

            // 2. prepare the layer with default hyper parameters 
            var imageClassifier = new Sequential(new Dictionary<string, object>
            {
                { "init", CNTKLib.GlorotUniformInitializer() },
                { "activation", Activation.Sigmoid },
                { "padding", true },
                { "strides", new int[] {2,2}},
                {"dropoutProb", 0.5 }
            });

            // 3. compose models with CNTK high level APIs  
            // create 3 layers of Convolution/dropout pairs end with one dense layer

            imageClassifier
                .Convolution(new int[] { 3, 3 }, 1)
                .Dropout()
                .Convolution(new int[] { 3, 3 }, 3)
                .Dropout()
                .Convolution(new int[] { 3, 3 }, 3)
                .Dropout()
                .Dense(outputDim);


            // 3.4. construct loss and prediction functions  
            Function lossFunction = CNTKLib.CrossEntropyWithSoftmax(imageClassifier, labels, "lossFunction");
            var prediction = CNTKLib.ClassificationError(imageClassifier, labels, "classificationError");
            Trainer trainer = Trainer.CreateTrainer(imageClassifier, lossFunction, prediction,
                new List<Learner>() { Learner.SGDLearner(imageClassifier.Parameters,
                new CNTK.TrainingParameterScheduleDouble(0.003125, TrainingParameterScheduleDouble.UnitType.Sample))});
        }

        //class MnistInstance
        //{
        //    [Dim(28, 28)]
        //    float[,] Img { get; set; }
        //}
        //class MnistLabel
        //{
        //    [Dim(10)]
        //    float[] Lbl { get; set; }
        //}

        //void Example2()
        //{
        //    // streams are the usual way to move stuff around
        //    var stream = File.Open("C:\foo.bin");

        //    // this will fail if the model does not match input-ouputs
        //    // async loading
        //    var model = Model.Load<MnistInstance, MnistLabel>(stream);

        //var v = new MnistInstance();

        //    // async execution
        //var label = model.Run(v); // MnistLabel
        //}
    }
}

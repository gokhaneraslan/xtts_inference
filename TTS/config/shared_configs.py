from dataclasses import asdict, dataclass
from typing import List

from TTS.coqpit import Coqpit, check_argument
from dataclasses import dataclass, field
from typing import Dict, List, Union


@dataclass
class BaseAudioConfig(Coqpit):
    """Base config to definge audio processing parameters. It is used to initialize
    ```TTS.utils.audio.AudioProcessor.```

    Args:
        fft_size (int):
            Number of STFT frequency levels aka.size of the linear spectogram frame. Defaults to 1024.

        win_length (int):
            Each frame of audio is windowed by window of length ```win_length``` and then padded with zeros to match
            ```fft_size```. Defaults to 1024.

        hop_length (int):
            Number of audio samples between adjacent STFT columns. Defaults to 1024.

        frame_shift_ms (int):
            Set ```hop_length``` based on milliseconds and sampling rate.

        frame_length_ms (int):
            Set ```win_length``` based on milliseconds and sampling rate.

        stft_pad_mode (str):
            Padding method used in STFT. 'reflect' or 'center'. Defaults to 'reflect'.

        sample_rate (int):
            Audio sampling rate. Defaults to 22050.

        resample (bool):
            Enable / Disable resampling audio to ```sample_rate```. Defaults to ```False```.

        preemphasis (float):
            Preemphasis coefficient. Defaults to 0.0.

        ref_level_db (int): 20
            Reference Db level to rebase the audio signal and ignore the level below. 20Db is assumed the sound of air.
            Defaults to 20.

        do_sound_norm (bool):
            Enable / Disable sound normalization to reconcile the volume differences among samples. Defaults to False.

        log_func (str):
            Numpy log function used for amplitude to DB conversion. Defaults to 'np.log10'.

        do_trim_silence (bool):
            Enable / Disable trimming silences at the beginning and the end of the audio clip. Defaults to ```True```.

        do_amp_to_db_linear (bool, optional):
            enable/disable amplitude to dB conversion of linear spectrograms. Defaults to True.

        do_amp_to_db_mel (bool, optional):
            enable/disable amplitude to dB conversion of mel spectrograms. Defaults to True.

        pitch_fmax (float, optional):
            Maximum frequency of the F0 frames. Defaults to ```640```.

        pitch_fmin (float, optional):
            Minimum frequency of the F0 frames. Defaults to ```1```.

        trim_db (int):
            Silence threshold used for silence trimming. Defaults to 45.

        do_rms_norm (bool, optional):
            enable/disable RMS volume normalization when loading an audio file. Defaults to False.

        db_level (int, optional):
            dB level used for rms normalization. The range is -99 to 0. Defaults to None.

        power (float):
            Exponent used for expanding spectrogra levels before running Griffin Lim. It helps to reduce the
            artifacts in the synthesized voice. Defaults to 1.5.

        griffin_lim_iters (int):
            Number of Griffing Lim iterations. Defaults to 60.

        num_mels (int):
            Number of mel-basis frames that defines the frame lengths of each mel-spectrogram frame. Defaults to 80.

        mel_fmin (float): Min frequency level used for the mel-basis filters. ~50 for male and ~95 for female voices.
            It needs to be adjusted for a dataset. Defaults to 0.

        mel_fmax (float):
            Max frequency level used for the mel-basis filters. It needs to be adjusted for a dataset.

        spec_gain (int):
            Gain applied when converting amplitude to DB. Defaults to 20.

        signal_norm (bool):
            enable/disable signal normalization. Defaults to True.

        min_level_db (int):
            minimum db threshold for the computed melspectrograms. Defaults to -100.

        symmetric_norm (bool):
            enable/disable symmetric normalization. If set True normalization is performed in the range [-k, k] else
            [0, k], Defaults to True.

        max_norm (float):
            ```k``` defining the normalization range. Defaults to 4.0.

        clip_norm (bool):
            enable/disable clipping the our of range values in the normalized audio signal. Defaults to True.

        stats_path (str):
            Path to the computed stats file. Defaults to None.
    """

    # stft parameters
    fft_size: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    frame_shift_ms: int = None
    frame_length_ms: int = None
    stft_pad_mode: str = "reflect"
    # audio processing parameters
    sample_rate: int = 22050
    resample: bool = False
    preemphasis: float = 0.0
    ref_level_db: int = 20
    do_sound_norm: bool = False
    log_func: str = "np.log10"
    # silence trimming
    do_trim_silence: bool = True
    trim_db: int = 45
    # rms volume normalization
    do_rms_norm: bool = False
    db_level: float = None
    # griffin-lim params
    power: float = 1.5
    griffin_lim_iters: int = 60
    # mel-spec params
    num_mels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = None
    spec_gain: int = 20
    do_amp_to_db_linear: bool = True
    do_amp_to_db_mel: bool = True
    # f0 params
    pitch_fmax: float = 640.0
    pitch_fmin: float = 1.0
    # normalization params
    signal_norm: bool = True
    min_level_db: int = -100
    symmetric_norm: bool = True
    max_norm: float = 4.0
    clip_norm: bool = True
    stats_path: str = None

    def check_values(
        self,
    ):
        """Check config fields"""
        c = asdict(self)
        check_argument("num_mels", c, restricted=True, min_val=10, max_val=2056)
        check_argument("fft_size", c, restricted=True, min_val=128, max_val=4058)
        check_argument("sample_rate", c, restricted=True, min_val=512, max_val=100000)
        check_argument(
            "frame_length_ms",
            c,
            restricted=True,
            min_val=10,
            max_val=1000,
            alternative="win_length",
        )
        check_argument("frame_shift_ms", c, restricted=True, min_val=1, max_val=1000, alternative="hop_length")
        check_argument("preemphasis", c, restricted=True, min_val=0, max_val=1)
        check_argument("min_level_db", c, restricted=True, min_val=-1000, max_val=10)
        check_argument("ref_level_db", c, restricted=True, min_val=0, max_val=1000)
        check_argument("power", c, restricted=True, min_val=1, max_val=5)
        check_argument("griffin_lim_iters", c, restricted=True, min_val=10, max_val=1000)

        # normalization parameters
        check_argument("signal_norm", c, restricted=True)
        check_argument("symmetric_norm", c, restricted=True)
        check_argument("max_norm", c, restricted=True, min_val=0.1, max_val=1000)
        check_argument("clip_norm", c, restricted=True)
        check_argument("mel_fmin", c, restricted=True, min_val=0.0, max_val=1000)
        check_argument("mel_fmax", c, restricted=True, min_val=500.0, allow_none=True)
        check_argument("spec_gain", c, restricted=True, min_val=1, max_val=100)
        check_argument("do_trim_silence", c, restricted=True)
        check_argument("trim_db", c, restricted=True)


@dataclass
class BaseDatasetConfig(Coqpit):
    """Base config for TTS datasets.

    Args:
        formatter (str):
            Formatter name that defines used formatter in ```TTS.tts.datasets.formatter```. Defaults to `""`.

        dataset_name (str):
            Unique name for the dataset. Defaults to `""`.

        path (str):
            Root path to the dataset files. Defaults to `""`.

        meta_file_train (str):
            Name of the dataset meta file. Or a list of speakers to be ignored at training for multi-speaker datasets.
            Defaults to `""`.

        ignored_speakers (List):
            List of speakers IDs that are not used at the training. Default None.

        language (str):
            Language code of the dataset. If defined, it overrides `phoneme_language`. Defaults to `""`.

        phonemizer (str):
            Phonemizer used for that dataset's language. By default it uses `DEF_LANG_TO_PHONEMIZER`. Defaults to `""`.

        meta_file_val (str):
            Name of the dataset meta file that defines the instances used at validation.

        meta_file_attn_mask (str):
            Path to the file that lists the attention mask files used with models that require attention masks to
            train the duration predictor.
    """

    formatter: str = ""
    dataset_name: str = ""
    path: str = ""
    meta_file_train: str = ""
    ignored_speakers: List[str] = None
    language: str = ""
    phonemizer: str = ""
    meta_file_val: str = ""
    meta_file_attn_mask: str = ""

    def check_values(
        self,
    ):
        """Check config fields"""
        c = asdict(self)
        check_argument("formatter", c, restricted=True)
        check_argument("path", c, restricted=True)
        check_argument("meta_file_train", c, restricted=True)
        check_argument("meta_file_val", c, restricted=False)
        check_argument("meta_file_attn_mask", c, restricted=False)



@dataclass
class TrainerConfig(Coqpit):

    # Fields for the run
    output_path: str = field(default="output")
    logger_uri: str = field(
        default=None,
        metadata={
            "help": "URI to save training artifacts by the logger. If not set, logs will be saved in the output_path. Defaults to None"
        },
    )
    run_name: str = field(default="run", metadata={"help": "Name of the run. Defaults to 'run'"})
    project_name: str = field(default=None, metadata={"help": "Name of the project. Defaults to None"})
    run_description: str = field(
        default="🐸Coqui trainer run.",
        metadata={"help": "Notes and description about the run. Defaults to '🐸Coqui trainer run.'"},
    )
    # Fields for logging
    print_step: int = field(
        default=25, metadata={"help": "Print training stats on the terminal every print_step steps. Defaults to 25"}
    )
    plot_step: int = field(
        default=100, metadata={"help": "Plot training stats on the logger every plot_step steps. Defaults to 100"}
    )
    model_param_stats: bool = field(
        default=False, metadata={"help": "Log model parameters stats on the logger dashboard. Defaults to False"}
    )
    wandb_entity: str = field(default=None, metadata={"help": "Wandb entity to log the run. Defaults to None"})
    dashboard_logger: str = field(
        default="tensorboard", metadata={"help": "Logger to use for the tracking dashboard. Defaults to 'tensorboard'"}
    )
    # Fields for checkpointing
    save_on_interrupt: bool = field(
        default=True, metadata={"help": "Save checkpoint on interrupt (Ctrl+C). Defaults to True"}
    )
    log_model_step: int = field(
        default=None,
        metadata={
            "help": "Save checkpoint to the logger every log_model_step steps. If not defined `save_step == log_model_step`."
        },
    )
    save_step: int = field(
        default=10000, metadata={"help": "Save local checkpoint every save_step steps. Defaults to 10000"}
    )
    save_n_checkpoints: int = field(default=5, metadata={"help": "Keep n local checkpoints. Defaults to 5"})
    save_checkpoints: bool = field(default=True, metadata={"help": "Save checkpoints locally. Defaults to True"})
    save_all_best: bool = field(
        default=False, metadata={"help": "Save all best checkpoints and keep the older ones. Defaults to False"}
    )
    save_best_after: int = field(
        default=0, metadata={"help": "Wait N steps to save best checkpoints. Defaults to 0"}
    )
    target_loss: str = field(
        default=None, metadata={"help": "Target loss name to select the best model. Defaults to None"}
    )
    # Fields for eval and test run
    print_eval: bool = field(default=False, metadata={"help": "Print eval steps on the terminal. Defaults to False"})
    test_delay_epochs: int = field(default=0, metadata={"help": "Wait N epochs before running the test. Defaults to 0"})
    run_eval: bool = field(
        default=True, metadata={"help": "Run evalulation epoch after training epoch. Defaults to True"}
    )
    run_eval_steps: int = field(
        default=None,
        metadata={
            "help": "Run evalulation epoch after N steps. If None, waits until training epoch is completed. Defaults to None"
        },
    )
    # Fields for distributed training
    distributed_backend: str = field(
        default="nccl", metadata={"help": "Distributed backend to use. Defaults to 'nccl'"}
    )
    distributed_url: str = field(
        default="tcp://localhost:54321",
        metadata={"help": "Distributed url to use. Defaults to 'tcp://localhost:54321'"},
    )
    # Fields for training specs
    mixed_precision: bool = field(default=False, metadata={"help": "Use mixed precision training. Defaults to False"})
    precision: str = field(
        default="fp16",
        metadata={
            "help": "Precision to use in mixed precision training. `fp16` for float16 and `bf16` for bfloat16. Defaults to 'f16'"
        },
    )
    epochs: int = field(default=1000, metadata={"help": "Number of epochs to train. Defaults to 1000"})
    batch_size: int = field(default=32, metadata={"help": "Batch size to use. Defaults to 32"})
    eval_batch_size: int = field(default=16, metadata={"help": "Batch size to use for eval. Defaults to 16"})
    grad_clip: float = field(
        default=0.0, metadata={"help": "Gradient clipping value. Disabled if <= 0. Defaults to 0.0"}
    )
    scheduler_after_epoch: bool = field(
        default=True,
        metadata={"help": "Step the scheduler after each epoch else step after each iteration. Defaults to True"},
    )
    # Fields for optimzation
    lr: Union[float, List[float]] = field(
        default=0.001, metadata={"help": "Learning rate for each optimizer. Defaults to 0.001"}
    )
    optimizer: Union[str, List[str]] = field(default=None, metadata={"help": "Optimizer(s) to use. Defaults to None"})
    optimizer_params: Union[Dict, List[Dict]] = field(
        default_factory=dict, metadata={"help": "Optimizer(s) arguments. Defaults to {}"}
    )
    lr_scheduler: Union[str, List[str]] = field(
        default=None, metadata={"help": "Learning rate scheduler(s) to use. Defaults to None"}
    )
    lr_scheduler_params: Dict = field(
        default_factory=dict, metadata={"help": "Learning rate scheduler(s) arguments. Defaults to {}"}
    )
    use_grad_scaler: bool = field(
        default=False,
        metadata={
            "help": "Enable/disable gradient scaler explicitly. It is enabled by default with AMP training. Defaults to False"
        },
    )
    allow_tf32: bool = field(
        default=False,
        metadata={
            "help": "A bool that controls whether TensorFloat-32 tensor cores may be used in matrix multiplications on Ampere or newer GPUs. Default to False."
        },
    )
    cudnn_enable: bool = field(default=True, metadata={"help": "Enable/disable cudnn explicitly. Defaults to True"})
    cudnn_deterministic: bool = field(
        default=False,
        metadata={
            "help": "Enable/disable deterministic cudnn operations. Set this True for reproducibility but it slows down training significantly.  Defaults to False."
        },
    )
    cudnn_benchmark: bool = field(
        default=False,
        metadata={
            "help": "Enable/disable cudnn benchmark explicitly. Set this False if your input size change constantly. Defaults to False"
        },
    )
    training_seed: int = field(
        default=54321,
        metadata={"help": "Global seed for torch, random and numpy random number generator. Defaults to 54321"},
    )



@dataclass
class BaseTrainingConfig(TrainerConfig):
    """Base config to define the basic 🐸TTS training parameters that are shared
    among all the models. It is based on ```Trainer.TrainingConfig```.

    Args:
        model (str):
            Name of the model that is used in the training.

        num_loader_workers (int):
            Number of workers for training time dataloader.

        num_eval_loader_workers (int):
            Number of workers for evaluation time dataloader.
    """

    model: str = None
    # dataloading
    num_loader_workers: int = 0
    num_eval_loader_workers: int = 0
    use_noise_augment: bool = False

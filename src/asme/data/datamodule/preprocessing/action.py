from abc import abstractmethod

from asme.core.init.context import Context

# TODO (AD): check handling of prefixes, i think that is overengineered
MAIN_FILE_KEY = "main_file"
SESSION_INDEX_KEY = "session_index"
OUTPUT_DIR_KEY = "output_dir"
PREFIXES_KEY = "prefixes"
DELIMITER_KEY = "delimiter"
SEED_KEY = "seed"
INPUT_DIR_KEY = "input_dir"
SPLIT_FILE_PREFIX = "split_file_prefix"
SPLIT_FILE_SUFFIX = "split_file_suffix"


class PreprocessingAction:
    """
    Base class for all actions that are performed by the AsmeDatamodule during preprocessing of dataset.
    """

    @abstractmethod
    def name(self) -> str:
        """
        The name of the preprocessing action. Used for logging progress.
        """
        pass

    def apply(self, context: Context, force_execution: bool = False) -> None:
        """
        Applies the preprocessing action. It can rely on information that has been saved in the context by previous
        actions and store data itself.

        :param context: Context that is preserved between actions. Actions can use the information to find new files
                        on-the-fly and hand on new data to down-stream actions.
        :param force_execution: If set to True, this disables dry-runs and forces each action to generate all files,
                                regardless of whether they are already available.
        """
        if not self.dry_run_available(context) or force_execution:
            self._run(context)
        else:
            self._dry_run(context)

    @abstractmethod
    def _dry_run(self, context: Context) -> None:
        """
        This should not generate new files, etc. but only populates the context with the information it would usually
        place there.
        """
        pass

    @abstractmethod
    def _run(self, context: Context) -> None:
        """
        This method should actually perform the preprocessing action. It should also populate the context with the
        information need the for stages that consume artifacts of this action.
        """
        pass

    @abstractmethod
    def dry_run_available(self, context: Context) -> bool:
        """
        Indicates whether a stage is capable of skipping generation of files if they are already present. It needs to be
        able to populate the context with the necessary keys for later stages nonetheless.
        """
        pass

    def __call__(self, context: Context, force_execution: bool = False) -> None:
        self.apply(context, force_execution)



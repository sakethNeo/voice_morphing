import tensorflow as tf
from keras import layers, Model
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import sys


def create_spectrogram_from_audio(
    file_path, target_shape=(128, 128), n_fft=2048, hop_length=512
):
    """
    Loads an audio file and converts it to a 2-channel spectrogram,
    returning the parameters used.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)  # Load audio file

        # Generate a Mel spectrogram using explicit parameters
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
        )

        # Convert to decibels (log scale)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # --- Handling the Shape and Channel Requirement ---
        # 1. Resize the spectrogram
        log_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, axis=-1)
        resized_spectrogram = tf.image.resize(log_mel_spectrogram, target_shape)

        # 2. Duplicate the single channel
        two_channel_spectrogram = tf.concat(
            [resized_spectrogram, resized_spectrogram], axis=-1
        )

        # Return the parameters along with the spectrogram
        return two_channel_spectrogram, sr, n_fft, hop_length

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None, None


# --- 1. Building Blocks of the U-Net (with naming for robustness) ---


def conv_block(inputs, num_filters, name):
    """Convolutional block with two Conv2D layers, now with naming."""
    x = layers.Conv2D(num_filters, 3, padding="same", name=f"{name}_conv1")(inputs)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(num_filters, 3, padding="same", name=f"{name}_conv2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x


def encoder_block(inputs, num_filters, name):
    """Encoder block: ConvBlock followed by MaxPooling, now with naming."""
    skip = conv_block(inputs, num_filters, name=f"{name}_conv_block")
    pool = layers.MaxPooling2D((2, 2), name=f"{name}_pool")(skip)
    return skip, pool


def decoder_block(inputs, skip_features, num_filters, name):
    """Decoder block: UpSampling, Concatenation, and ConvBlock, now with naming."""
    x = layers.UpSampling2D((2, 2), name=f"{name}_upsample")(inputs)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip_features])
    x = conv_block(x, num_filters, name=f"{name}_conv_block")
    return x


# --- 2. Defining the U-Net and Decoder Models ---


def build_simple_unet(input_shape):
    """Builds the U-Net model and a multi-output Encoder."""
    inputs = layers.Input(input_shape)

    # Encoder Path
    s1, p1 = encoder_block(inputs, 64, name="encoder_1")
    s2, p2 = encoder_block(p1, 128, name="encoder_2")
    s3, p3 = encoder_block(p2, 256, name="encoder_3")

    # Bottleneck
    bottleneck = conv_block(p3, 512, name="bottleneck")

    # Decoder Path
    d3 = decoder_block(bottleneck, s3, 256, name="decoder_3")
    d2 = decoder_block(d3, s2, 128, name="decoder_2")
    d1 = decoder_block(d2, s1, 64, name="decoder_1")

    # Output Layer
    outputs = layers.Conv2D(
        1, 1, padding="same", activation="sigmoid", name="output_layer"
    )(d1)

    # Create the full U-Net model
    model = Model(inputs, outputs, name="SimpleUNet")

    # Create an Encoder model that outputs the bottleneck and all skip connections
    encoder_model = Model(inputs, outputs=[bottleneck, s1, s2, s3], name="Encoder")

    return model, encoder_model


def build_decoder(latent_shape, skip_shapes):
    """Builds a standalone Decoder model."""
    latent_input = layers.Input(shape=latent_shape, name="latent_input")
    s1_input = layers.Input(shape=skip_shapes[0], name="s1_input")
    s2_input = layers.Input(shape=skip_shapes[1], name="s2_input")
    s3_input = layers.Input(shape=skip_shapes[2], name="s3_input")

    # Decoder path using the same architecture and names as the U-Net
    d3 = decoder_block(latent_input, s3_input, 256, name="decoder_3")
    d2 = decoder_block(d3, s2_input, 128, name="decoder_2")
    d1 = decoder_block(d2, s1_input, 64, name="decoder_1")

    decoder_outputs = layers.Conv2D(
        1, 1, padding="same", activation="sigmoid", name="output_layer"
    )(d1)

    # Create the model
    decoder_model = Model(
        inputs=[latent_input, s1_input, s2_input, s3_input],
        outputs=decoder_outputs,
        name="Decoder",
    )
    return decoder_model


# --- 3. Example Usage ---

if __name__ == "__main__":
    input_shape = (128, 128, 2)
    unet_model, encoder_model = build_simple_unet(input_shape)
    print("--- U-Net Model Summary ---")
    unet_model.summary()

    # --- Latent Space Interpolation ---

    input_shape = (128, 128, 2)
    unet_model, encoder_model = build_simple_unet(input_shape)

    audio_file_A = "C:/Users/ambal/Downloads/ESC-50-master/ESC-50-master/audio/2-102414-A-17.wav"  # Example: Water
    audio_file_B = "C:/Users/ambal/Downloads/ESC-50-master/ESC-50-master/audio/2-101676-A-10.wav"  # Example: Rain

    spectrogram_A_tensor, sr_A, n_fft_param, hop_length_param = (
        create_spectrogram_from_audio(audio_file_A)
    )
    spectrogram_B_tensor, _, _, _ = create_spectrogram_from_audio(audio_file_B)

    # The model expects a batch dimension, so we add one.
    spectrogram_pair_A = np.expand_dims(spectrogram_A_tensor.numpy(), axis=0)
    spectrogram_pair_B = np.expand_dims(spectrogram_B_tensor.numpy(), axis=0)

    # 2. Encode inputs to get latent vectors and skip connections
    latent_A, s1_A, s2_A, s3_A = encoder_model.predict(spectrogram_pair_A)
    latent_B, s1_B, s2_B, s3_B = encoder_model.predict(spectrogram_pair_B)

    # 3. Interpolate in the latent space
    alpha = 0.5  # Midpoint
    interpolated_latent = alpha * latent_A + (1 - alpha) * latent_B
    interpolated_s1 = alpha * s1_A + (1 - alpha) * s1_B
    interpolated_s2 = alpha * s2_A + (1 - alpha) * s2_B
    interpolated_s3 = alpha * s3_A + (1 - alpha) * s3_B

    # 4. Build the standalone decoder
    latent_shape = encoder_model.outputs[0].shape[1:]
    skip_shapes = [
        encoder_model.outputs[1].shape[1:],
        encoder_model.outputs[2].shape[1:],
        encoder_model.outputs[3].shape[1:],
    ]
    decoder_model = build_decoder(latent_shape, skip_shapes)

    # 5. Copy weights from the U-Net to the Decoder
    # This is crucial for the decoder to function correctly.
    for layer in decoder_model.layers:
        if layer.get_weights():  # Only for layers with weights
            unet_layer = unet_model.get_layer(layer.name)
            layer.set_weights(unet_layer.get_weights())

    generated_spectrogram = decoder_model.predict(
        [interpolated_latent, interpolated_s1, interpolated_s2, interpolated_s3]
    )

    print("\n--- Latent Space Interpolation Success ---")
    print(f"Shape of Latent Vector A: {latent_A.shape}")
    print(f"Shape of Interpolated Latent: {interpolated_latent.shape}")
    print(f"Shape of Generated Spectrogram: {generated_spectrogram.shape}")

    print("\n--- Plotting Results ---")

    # We only need one channel to visualize the spectrogram
    plot_spec_A = spectrogram_pair_A[0, :, :, 0]
    plot_spec_B = spectrogram_pair_B[0, :, :, 0]
    generated_output = generated_spectrogram[0, :, :, 0]

    # Get the min and max dB values from an input spectrogram
    min_db = np.min(plot_spec_A)
    max_db = np.max(plot_spec_A)

    # Rescale the generated output from [0, 1] to [min_db, max_db]
    plot_spec_generated = generated_output * (max_db - min_db) + min_db

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.ion()
    # Plot Spectrogram A
    img_A = librosa.display.specshow(
        plot_spec_A, ax=axes[0], x_axis="time", y_axis="mel", vmin=min_db, vmax=max_db
    )
    axes[0].set_title("Input Spectrogram A")
    fig.colorbar(img_A, ax=axes[0], format="%+2.0f dB")

    # Plot Generated Spectrogram
    img_gen = librosa.display.specshow(
        plot_spec_generated,
        ax=axes[1],
        x_axis="time",
        y_axis="mel",
        vmin=min_db,
        vmax=max_db,
    )
    axes[1].set_title("Generated (Interpolated)")
    fig.colorbar(img_gen, ax=axes[1], format="%+2.0f dB")

    # Plot Spectrogram B
    img_B = librosa.display.specshow(
        plot_spec_B, ax=axes[2], x_axis="time", y_axis="mel", vmin=min_db, vmax=max_db
    )
    axes[2].set_title("Input Spectrogram B")
    fig.colorbar(img_B, ax=axes[2], format="%+2.0f dB")

    plt.show(block=False)
    plt.pause(0.001)

    print("\n--- Converting generated spectrogram to audio ---")

    sr = sr_A
    n_fft = n_fft_param
    hop_length = hop_length_param

    if sr is None:
        print("Error: Sample rate not found. Cannot proceed with audio generation.")
    else:
        try:
            # 1. Convert the generated spectrogram from dB scale back to power
            print("1/3: Converting spectrogram from dB to power scale...")
            sys.stdout.flush()  # Force print to appear immediately
            generated_power_spectrogram = librosa.db_to_power(plot_spec_generated)

            # 2. Invert the Mel spectrogram to get the audio waveform
            print(
                "2/3: Reconstructing audio with Griffin-Lim... (This may take a moment)"
            )
            sys.stdout.flush()

            generated_waveform = librosa.feature.inverse.mel_to_audio(
                generated_power_spectrogram, sr=sr, n_fft=n_fft, hop_length=hop_length
            )

            # 3. Save the generated waveform to a file
            print("3/3: Saving the audio file...")
            sys.stdout.flush()
            output_filename = "generated_interpolated_audio.wav"
            sf.write(output_filename, generated_waveform, sr)

            print(f"\n✅ Successfully saved the generated audio to: {output_filename}")

        except Exception as e:
            print(f"\n❌ An error occurred during audio conversion: {e}")
            print(
                "This often happens if the spectrogram parameters (n_fft, hop_length) for creation and inversion do not match."
            )
